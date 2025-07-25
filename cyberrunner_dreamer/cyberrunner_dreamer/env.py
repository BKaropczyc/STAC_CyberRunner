from cyberrunner_dreamer import cyberrunner_layout
from cyberrunner_dreamer.path import LinearPath
from cyberrunner_dreamer.layout_renderer import LayoutRenderer

from cyberrunner_interfaces.msg import DynamixelVel, StateEstimateSub
from cyberrunner_interfaces.srv import DynamixelReset

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import gym
import numpy as np
import time
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from typing import Optional


class CyberrunnerGym(gym.Env):
    metadata = {"render_modes": ["human", "single_rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        repeat=1,
        layout=cyberrunner_layout.cyberrunner_hard_layout,
        num_rel_path=5,
        num_wait_steps=30,
        reward_on_fail=0.0,
        reward_on_goal=0.5,
        send_actions=True
    ):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()

        # Define the Observation and Action spaces for the environment
        # Observation space
        # i.e., what the agent "sees" from the environment at each time step
        self.observation_space = gym.spaces.Dict(
            image=gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),  # The maze image around the ball
            states=gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),   # alpha, beta, xb, yb
            goal=gym.spaces.Box(-np.inf, np.inf, (num_rel_path * 2,), np.float32),  # Direction to the goal along the path
            progress=gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),  # Progress along the path
            log_reward=gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        )

        # Action space
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,))  # A relative current command for each servo

        # Initialize our instance variables
        self.render_mode = render_mode
        self.br = CvBridge()
        self.repeat = repeat     # The number of times our ROS node will 'spin' to get each new observation
        self.offset = np.array([0.276, 0.231]) / 2.0    # Move the origin for ball positions to the lower-left corner of the maze
                                                        # TODO: Check these measurements!
        shared = get_package_share_directory("cyberrunner_dreamer")
        self.p = LinearPath.load(os.path.join(shared, "path_0002_hard.pkl"))
        self.renderer = LayoutRenderer(layout, scale=1500)

        self.prev_path_pos = 0                  # The most recent position achieved along the path to the goal
        self.num_wait_steps = num_wait_steps    # Number of steps to wait before starting a new episode
        self.reward_on_fail = reward_on_fail    # Additional reward when reaching a failure state
        self.reward_on_goal = reward_on_goal    # Additional reward for reaching the goal state

        self.ball_detected = False              # Whether the ball was detected in the most recent observation

        # Dynamixel Limits
        self.max_current = 200      # Max goal current to send to the servos
        self.alpha_fac = -1.0       # Scaling factor for servo 0 current
        self.beta_fac = -1.0        # Scaling factor for servo 1 current

        self.episodes = 0        # Number of total episodes started
        self.steps = 0           # Number of steps taken in the current episode
        self.accum_reward = 0.0  # The sum of all rewards in the current episode
        self.last_time = 0       # The last time we completed a step
        self.progress = 0        # The progress made toward the goal from the most recent action
        self.success = False     # Did we achieve the goal state?
        self.off_path = False    # Is the marble too far off the specified path?
        self.new_obs = False     # Semaphore to signal the receipt of a new ROS state estimation message

        self.cheat = False       # Whether the agent is allowed to cheat
        self.obs = dict(self.observation_space.sample())    # The most recent (valid) observation
        self.num_rel_path = num_rel_path    # Number of points along the path to include in the "goal" field
        self.send_actions = send_actions   # Set False to disable sending messages to the servos, for debugging

        # Set up normalization constants for our state and goal observations
        # Incoming observations will be divided by these to yield "relative" observations
        self.norm_max = np.array([np.deg2rad(10), np.deg2rad(10), 0.276, 0.231])
        self.goal_norm_max = np.array(
            [0.0002 * 60 * k for k in range(1, self.num_rel_path + 1) for _ in range(2)]
        )

        # Create our ROS node
        self.node = Node("cyberrunner_gym")

        # Listen for state estimation messages...
        self.subscription = self.node.create_subscription(
            StateEstimateSub,
            topic="cyberrunner_state_estimation/estimate_subimg",
            callback=self._msg_to_obs,
            qos_profile=1
        )

        # Publish messages to the Dynamixel node
        self.publisher = self.node.create_publisher(
            DynamixelVel,
            topic="cyberrunner_dynamixel/cmd",
            qos_profile=1
        )

        # Prepare to call the Dynamixel's 'reset' service
        self.client = self.node.create_client(
            DynamixelReset, "cyberrunner_dynamixel/reset"
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Gym environment primary methods
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self, *, seed=None, return_info=False, options=None):
        """
        Reset the environment to begin a new episode.
        """
        print("Resetting the environment...")
        super().reset(seed=seed)

        # Reset / update instance vars
        self.episodes += 1
        self.steps = 0
        self.accum_reward = 0.0
        self.progress = 0
        self.success = False
        self.ball_detected = False
        self.off_path = False
        self.prev_path_pos = 0

        # Reset the board
        self._reset_board()

        # Wait until we are ready to start the next episode
        kb_reset = False
        if kb_reset:
            # Wait for a keyboard press
            # TODO: Show a better picture!!!
            win_name = "Waiting for a keypress..."
            cv2.imshow(win_name, np.zeros((200, 200, 3)))
            cv2.waitKey(0)
            cv2.destroyWindow(win_name)

            # Get the first observation to return
            obs = self._get_obs()

        else:
            # Wait until the ball appears near the "start" location
            print("Waiting for the ball to appear...")

            # Make sure we get num_wait_steps consecutive frames with the ball in the starting location
            starting_pos = self.p.points[0]
            count = 0
            while count < self.num_wait_steps:
                # Check if the ball is present near the starting location
                ball_at_start = False
                obs = self._get_obs()
                if self.ball_detected:
                    state = obs["states"]
                    ball_pos = np.array([state[2], state[3]])
                    if np.linalg.norm(ball_pos - starting_pos) < 0.01:
                        ball_at_start = True

                # Update our counter
                if ball_at_start:
                    count += 1
                else:
                    count = 0    # Reset the counter

                # Keep OpenCV processing events (in case we have render windows open)
                cv2.waitKey(1)
        print(f"Playing episode {self.episodes}...")

        # Initialize prev_path_pos
        self.prev_path_pos = self.p.closest_point(obs["states"][2:4])[0]
        self.last_time = time.time()

        # Finalize and return the observation
        self._normalize_obs(obs)
        obs["progress"] = np.asarray([1 + self.prev_path_pos], dtype=np.float32)
        obs["log_reward"] = np.array([0], dtype=np.float32)

        if return_info:
            info_dict = {}   # Currently unused
            return obs, info_dict
        else:
            return obs

    def step(self, action):
        """Take an action in the environment and return the observation and reward."""

        # We've taken one more step this episode
        self.steps += 1

        # Send action to Dynamixel
        # If we can cheat, and we're close to the goal...
        if self.cheat and (self.p.num_points - self.prev_path_pos <= 200):
            # Force a tilt toward the goal
            action[1] = 1.0
        self._send_action(action)

        # Get the next observation
        obs = self._get_obs()

        # Compute the progress along the path and the associated reward
        reward = self._get_reward(obs)

        # Determine whether the episode is now over
        done, reason = self._get_done(obs)
        if done:
            # Stop the servos
            self._send_action(np.zeros(2))

            # Update the reward
            if self.success:
                reward += self.reward_on_goal
            else:
                reward = self.reward_on_fail

        # Update the accumulated reward for this episode
        self.accum_reward += reward

        # Summarize the episode if we are done
        if done:
            print(f"Finished episode {self.episodes}:")
            print(f"  Reason: {reason}")
            print(f"  Episode length: {self.steps} steps")
            print(f"  Total accumulated reward: {self.accum_reward:0.4f}")

        # Output a warning if we're stepping too slowly
        now = time.time()
        if self.last_time and ((now - self.last_time) > (1.0 / 35.0)):
            print("WARNING: stepping slower than 35fps")
        self.last_time = now

        # Define our "info" dictionary
        if self.success:
            info = {"is_terminal": False}
        else:
            info = {}

        # Finalize the observation
        self._normalize_obs(obs)
        obs["progress"] = np.asarray([1 + self.prev_path_pos], dtype=np.float32)
        obs["log_reward"] = np.asarray([reward if not done else 0], dtype=np.float32)

        # Return the step results
        return obs, reward, done, info

    def render(self, mode="human"):
        # NOTE: Keyword arg mode is deprecated and will be removed in later versions.

        mode = self.render_mode
        if mode == "single_rgb_array" or mode == "human":
            # Determine the current call position
            ball_pos = None
            if self.ball_detected:
                ball_pos = self.obs["states"][2:]

            # Determine the closest path point
            path_pos = None
            if self.prev_path_pos >= 0:
                path_pos = self.p.points[self.prev_path_pos]

            # Get a visualization of the current state of the system
            frame = self.renderer.get_image(ball_pos, self.off_path, path_pos)

            if mode == "human":
                cv2.imshow("Layout", frame)
                cv2.waitKey(1)
            else:
                return frame
        elif mode is None:
            return
        else:
            super().render(mode=mode)  # Raise an exception

    # ------------------------------------------------------------------------------------------------------------------
    # Private utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def _reset_board(self):
        # Send a request to put the playing surface in a starting position
        if self.send_actions:
            req = DynamixelReset.Request()
            future = self.client.call_async(req)

            # Wait for board reset to be finished
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5)

    def _send_action(self, action):
        # Scale the action
        action = action.copy()
        action *= self.max_current
        vel_1 = self.alpha_fac * action[0]  # TODO define these as parameters
        vel_2 = self.beta_fac * action[1]

        # Convert to message and publish
        msg = DynamixelVel()
        msg.vel_1 = vel_1
        msg.vel_2 = vel_2
        if self.send_actions:
            self.publisher.publish(msg)

    def _get_obs(self):
        # Until we have a new observation...
        while not self.new_obs:
            # Spin repeat times to get the next observation
            for _ in range(self.repeat):
                rclpy.spin_once(self.node)
        self.new_obs = False

        # Return the new observation
        return self.obs.copy()

    def _msg_to_obs(self, msg):
        """
        ROS message to Gym observation
        NOTES:
        - This is NOT a complete observation. Additional keys need to be added by subsequent processing.
        - If the ball was not detected, we set self.ball_detected to False, but otherwise do NOT update
          the latest observation. In general this shouldn't be a problem because the episode is over anyway.
        """
        # Update ball_detected
        self.ball_detected = np.isfinite(msg.state.x_b)
        if self.ball_detected:
            # State vector
            states = np.array([msg.state.alpha, msg.state.beta, msg.state.x_b, msg.state.y_b])
            states[2:] += self.offset    # Convert to an origin in the lower-left corner of the maze

            # Path toward goal
            rel_path = self.p.get_rel_path(states[2:], self.num_rel_path, 60).flatten()

            # Ball subimage
            img = self.br.imgmsg_to_cv2(msg.subimg)

            # Store this as our current observation
            self.obs = {"states": states, "goal": rel_path, "image": img}

        # Indicate that we generated a new observation
        self.new_obs = True

    def _normalize_obs(self, obs):
        """Normalize the observation values"""
        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)

    # ------------------------------------------------------------------------------------------------------------------
    # Step processing utils
    # ------------------------------------------------------------------------------------------------------------------

    def _get_reward(self, obs):
        """Calculate and return the reward for the most recent action-observation step."""

        # Get the progress along the path toward the goal
        progress = self._get_progress(obs)

        # The reward is a scaled version of the progress
        reward = float(progress) * 0.004 / 16.0   # Where does this number come from???

        # Return the calculated reward
        return reward

    def _get_progress(self, obs):
        """Calculate and return the progress along the path made during the most recent action-observation step."""

        # If the ball is not detected, we can't calculate the progress
        if not self.ball_detected:
            progress = 0
            # Should we update prev_path_pos in this condition (to -1) ?
            # i.e., should prev_path_pos reflect the last KNOWN path position,
            # or the path position on the last step, even if it is unknown (off path)?

        else:
            # Determine the path point closest to the ball
            curr_path_pos, _ = self.p.closest_point(obs["states"][2:4])

            # See if we're off the path
            self.off_path = curr_path_pos == -1

            # Calculate our progress
            # If either the current or previous positions was "off path", we can't calculate the progress
            if self.off_path or (self.prev_path_pos == -1):
                progress = 0

            else:
                # The progress is the difference between the current and previous path points.
                progress = curr_path_pos - self.prev_path_pos

            # Remember the previous path position
            if self.off_path and self.cheat:
                pass    # If we can cheat, assume we are never off path. Just keep our old prev_path_pos
            else:
                self.prev_path_pos = curr_path_pos

        # Save and return the result
        self.progress = progress
        return progress

    def _get_done(self, obs):
        """
        Determine if the current episode has finished, and if so, why.

        Args:
            obs: The most recent environment observation

        Returns:
            done: bool, True if the episode has finished.

            reason: str, The reason the episode has finished if done is True
        """
        # Prepare to return our response
        reason = None

        # Done if ball is not detected
        if not self.ball_detected:
            reason = "BALL NOT DETECTED"

        # Done if we reached the goal
        elif self.p.num_points - self.prev_path_pos <= 1:
            reason = "SUCCESS!!!"
            self.success = True

        # Done if off path and not cheating
        elif (not self.cheat) and self.off_path:
            reason = "OFF PATH"

        # Done if we made too much progress in a single step
        elif (not self.cheat) and (self.progress > 300):
            reason = "TOO MUCH PROGRESS (cheated)"

        # Done if we've taken too many steps
        elif self.steps >= 3000:
            reason = "MAX STEPS REACHED"

        # We are 'done' if we defined a reason
        done = bool(reason)

        # Return our results
        return done, reason
