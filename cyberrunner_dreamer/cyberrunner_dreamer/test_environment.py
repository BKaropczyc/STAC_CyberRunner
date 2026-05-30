import gymnasium as gym
import cyberrunner_dreamer    # To register the environment with Gym

def main():
    # Initialize our environment
    env = gym.make("cyberrunner_dreamer:cyberrunner-ros-v0",
                   render_mode="human",
                   send_actions=False)
    env.action_space.seed(42)

    # Start a basic environment-interaction loop
    while True:
        print("Starting a new episode...")
        obs, info = env.reset()

        # Play one episode
        while True:
            # Take a random action
            act = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(act)

            if terminated or truncated:
                break   # End of this episode

    env.close()


if __name__ == "__main__":
    main()
