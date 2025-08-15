from datetime import datetime
import rclpy
from dreamerv3.train import main as train


def main(args=None):
    rclpy.init(args=args)

    # Define where we should save our logs
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    log_dir = "~/cyberrunner_logs/" + date_str      # Set to a constant to re-use old state

    # Prepare the parameters for our training loop
    argv = [
        "--configs", "cyberrunner", "large",  # TODO add config file here!
        "--logdir", log_dir,
        "--replay_size", "1e6",
        "--run.script", "train",
        "--run.train_ratio", "128",
        "--run.save_every", "20",
        "--run.log_every", "10",
        "--jax.policy_devices", "0",
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
