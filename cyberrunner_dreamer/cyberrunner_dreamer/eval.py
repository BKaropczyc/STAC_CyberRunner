from datetime import datetime
import rclpy
from dreamerv3.train import main as train


def main(args=None):
    rclpy.init(args=args)

    # Define where we should save our logs
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    logdir = "robust_2"
    checkpoint = f"~/cyberrunner_logs/{logdir}/checkpoint.ckpt"

    # Prepare the parameters for our evaluation loop
    argv = [
        "--configs", "cyberrunner", "large",  # TODO add config file here!
        "--task", "gym_cyberrunner_dreamer:cyberrunner-ros-v0",
        "--logdir", "~/cyberrunner_logs/eval_" + date_str,
        "--run.from_checkpoint", checkpoint,
        "--run.steps", "10000",
        "--run.script", "eval_only",
        "--jax.policy_devices", "0",
        "--jax.train_devices", "0"
    ]

    # Begin the evaluation
    train(argv)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
