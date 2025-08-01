from datetime import datetime
import rclpy
from dreamerv3.train import main as train


def main(args=None):
    rclpy.init(args=args)

    # Define where we should save our logs
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")    # Set to a constant to re-use old state

    # Prepare the parameters for our training loop
    argv = [
        "--configs", "cyberrunner", "large",  # TODO add config file here!
        "--task", "gym_cyberrunner_dreamer:cyberrunner-ros-v0",
        "--logdir", "~/cyberrunner_logs/" + date_str,
        "--replay_size", "1e6",
        "--run.script", "parallel",
        "--run.train_ratio", "-1",
        "--run.save_every", "20",
        "--run.log_every", "10",
        "--jax.policy_devices", "1",
        "--jax.train_devices", "0"
    ]

    # Restrict the training to select GPUs, if necessary:
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Begin the training
    train(argv)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
