import gym
import cyberrunner_dreamer    # To register the environment with Gym

# Ignore Gym API deprecation warnings
# We know we're using an older version of Gym
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message=".* old step API which returns one bool instead of two.*")

def main():
    # Initialize our environment
    env = gym.make("cyberrunner_dreamer:cyberrunner-ros-v0", new_step_api=False)
    env.action_space.seed(42)

    # Start a basic environment-interaction loop
    while True:
        print("Starting a new episode...")
        obs, info = env.reset(return_info=True)
        env.render()

        # Play one episode
        while True:
            # Take a random action
            act = env.action_space.sample()
            obs, reward, done, info = env.step(act)
            env.render()

            if done:
                break   # End of this episode

    env.close()


if __name__ == "__main__":
    main()
