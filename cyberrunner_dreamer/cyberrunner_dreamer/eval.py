from datetime import datetime
import rclpy
from dreamerv3.train import main as train


def main(args=None):
    rclpy.init(args=args)

    # Define where we should save our logs
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    log_dir = "~/cyberrunner_logs/eval_" + date_str

    # Determine which policy (i.e., checkpoint) to evaluate
    checkpoint = "~/cyberrunner_logs/Training_run_1/checkpoint.ckpt"     # UPDATE as necessary

    # Prepare the parameters for our evaluation loop
    argv = [
        "--configs", "cyberrunner", "large",  # TODO add config file here!
        "--logdir", log_dir,
        "--run.script", "eval_only",
        "--run.from_checkpoint", checkpoint,
        "--run.eval_eps", "20",
        "--jax.policy_devices", "0",
        "--jax.train_devices", "0"
    ]

    # Begin the evaluation
    train(argv)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
