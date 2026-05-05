import sys
from pathlib import Path
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory
from dreamerv3.embodied import Config
from dreamerv3.embodied.replay.saver import Saver
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from matplotlib.animation import FuncAnimation


def main():
    # Show a histogram of the "progress" values of the steps in the selected replay buffer

    # Initialize Tkinter and hide the main window
    root = Tk()
    root.withdraw()  # Hides the main window

    # Have the user pick the log directory whose replay buffer they would like to visualize
    selected_dir = askdirectory(title="Please select a CyberRunner log directory:",
                                initialdir="~/cyberrunner_logs",
                                mustexist=True)
    if not selected_dir:
        print("ERROR: No directory selected!", file=sys.stderr)
        return
    log_dir = Path(selected_dir)

    # Read in the config.yaml file
    config_file = log_dir / "config.yaml"
    if not config_file.is_file():
        print("ERROR: Could not find a \"config.yaml\" file in the selected directory!", file=sys.stderr)
        return
    config = Config.load(config_file)

    # Get the replay directory
    replay_dir = log_dir / "replay"
    if not replay_dir.is_dir():
        print("ERROR: Could not find the \"replay\" directory!", file=sys.stderr)
        return

    # Read in the original (non-augmented) data from the replay buffer
    orig_steps = list()     # The complete list of non-augmented training steps, in order
    episode_steps = list()  # The step count at the end of each episode (should match the scores.json file)
    all_streams = set()     # Set of all streams IDs (should be one per data augmentation variation per training restart)
    orig_streams = set()    # The stream ID's that correspond to original, non-augmented training steps

    print(f"Reading in the replay buffer...")
    saver = Saver(replay_dir)

    # For every step in the replay buffer (in chunk filename order)...
    for step, stream in saver.load(capacity=None, length=config.batch_length):
        # If this is a new stream ID...
        if stream not in all_streams:
            all_streams.add(stream)
            # Every 4th stream is assumed to contain original (i.e., non-augmented) experience data
            # During training, each variation of augmented data is written to a separate stream,
            # with the first stream corresponding the original data from the environment.
            if len(all_streams) % 4 == 1:
                orig_streams.add(stream)

        # If this step is from an original stream, collect it
        if stream in orig_streams:
            orig_steps.append(step)

            # If this step is the end of an episode, record the step number
            if step["is_last"]:
                episode_steps.append(len(orig_steps))

    print(f"Total number of original (non-augmented) steps in the replay buffer: {len(orig_steps):,}")
    num_episodes = len(episode_steps)
    print(f"Number of episodes in the replay buffer: {num_episodes:,}")

    # Validate the data
    print("Validating replay buffer data...")
    if len(all_streams) %4 != 0:
        print("ERROR: Number of streams loaded in was not a multiple of 4!", file=sys.stderr)
        return
    if orig_steps[0]["is_first"] != True:
        print("ERROR: First original step was not is_first=True", file=sys.stderr)
    if orig_steps[-1]["is_last"] != True:
        print("ERROR: Last original step was not is_last=True", file=sys.stderr)
    if orig_steps[-1]["is_terminal"] != True:
        print("ERROR: Last original step was not is_terminal=True", file=sys.stderr)
    num_starts = sum(1 for step in orig_steps if step["is_first"])
    num_lasts = sum(1 for step in orig_steps if step["is_last"])
    if num_starts != num_lasts:
        print(f"ERROR: Partial episodes detected! Number of is_first steps ({num_starts:,}) != number of is_last steps ({num_lasts:,})", file=sys.stderr)
        return
    if num_starts != num_episodes:
        print(f"ERROR: Number of is_first steps ({num_starts:,}) != number of episodes ({num_episodes:,})!", file=sys.stderr)
        return
    print("Validation succeeded!")

    # Plot the data in the replay buffer
    print("Plotting replay buffer distribution history...")
    assumed_replay_size = int(config.replay_size / 4)   # Only 1/4 of the replay buffer was used for original (non-augmented) data
    print(f" (assuming a replay capacity of {assumed_replay_size:,} non-augmented steps)")

    # NOTE: We don't plot the "terminal" steps in an episode since we never "learn" from them (they have no associated action)
    # Also, the "progress" value of the terminal steps is often very high due to "cheating" that ended the episode
    # We're trying to visualize what parts of the maze DreamerV3 is learning from
    vals = [np.nan if s["is_terminal"] else s["progress"].item() for s in orig_steps]
    num_bins = 100

    # 1. Set up the initial plot
    n, bins, patches = plt.hist([], bins=num_bins, range=(0, 9860), color="tab:red")
    plt.suptitle("Distribution of \"progress\" Values in the Replay Buffer", fontsize=14)
    title_text = plt.title("")
    plt.xlabel("Progress Along Maze")
    plt.ylabel("% of Replay Buffer Data")
    plt.ylim((0, 0.5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set_axisbelow(True)
    plt.grid(axis='y', alpha=0.3)

    # 2. Update function
    def update_plot(step_num):
        # Get the data for this frame
        data = vals[max(0, step_num - assumed_replay_size) : step_num]   # The replay buffer will have at most capacity / 4 original steps in it
        num_vals = np.count_nonzero(~np.isnan(data))
        # Compute new histogram values
        new_n, _ = np.histogram(data, bins=bins, weights=np.ones(len(data)) / num_vals)
        # Update rectangle heights
        for rect, h in zip(patches, new_n):
            rect.set_height(h)
        # Update the title
        title_text.set_text(f"(after {step_num:,} steps)")

        return (title_text,) + patches

    # 3. Create and show animation
    ani = FuncAnimation(plt.gcf(), update_plot, frames=episode_steps, blit=False, interval=100, repeat_delay=1000)
    plt.show()


if __name__ == "__main__":
    main()
