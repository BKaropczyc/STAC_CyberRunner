import sys
from collections import deque
from itertools import pairwise
from pathlib import Path
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory, asksaveasfilename
from dreamerv3.embodied import Config
from dreamerv3.embodied.replay.saver import Saver
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from matplotlib.animation import FuncAnimation

class HistogramQueue:
    """A class for efficiently maintaining histogram data for a collection of values
    as new data is added to the collection. Rather than re-calculating the entire
    histogram after new data is added, the counts for the new added are added to existing counts."""
    def __init__(self, range, bins, maxlen=None):
        self.range = range     # The lower and upper range of the histogram bins
        self.bins = bins       # The number of bins in the histogram
        self.maxlen = maxlen   # Only the most recent maxlen data items will be used for the histogram values
                               # This allows you to determine the distribution of just "recent" data

        self.bin_edges = np.histogram_bin_edges([], bins=bins, range=range)
        self._counts = np.zeros(bins, dtype=int)
        self._data = deque(maxlen=maxlen)   # A FIFO queue of only "recent" data (defined by maxlen)


    def add(self, new_data):
        """Add new data to the queue.
        After maxlen values have been added, old data is discarded and is not included in the histogram results."""
        if self.maxlen and len(new_data) >= self.maxlen:
            # data will completely replace our queue
            if len(new_data) > self.maxlen:
                # Trim data to the latest maxlen elements
                new_data = new_data[-self.maxlen:]
            self._data = deque(new_data, maxlen=self.maxlen)
            self._counts, _ = np.histogram(new_data, bins=self.bin_edges)

        else:
            # At least some of the existing data will remain
            num_overflow = len(self._data) + len(new_data) - self.maxlen if self.maxlen else 0
            if num_overflow > 0:
                # We will overflow our maxlen by adding this data
                # Remove the data that will overflow
                removed_items = [self._data.popleft() for _ in range(num_overflow)]
                remove_counts, _ = np.histogram(removed_items, bins=self.bin_edges)
                self._counts -= remove_counts

            # We can add this data without overflowing
            self._data.extend(new_data)
            data_counts, _ = np.histogram(new_data, bins=self.bin_edges)
            self._counts += data_counts


    def counts(self, as_proportion:bool=False):
        """Return the current histogram counts, optionally as a proportion of all the (recent) data."""
        if as_proportion:
            return (self._counts / np.sum(self._counts)).copy()
        else:
            return self._counts.copy()


class HistogramHistory(HistogramQueue):
    """
    A subclass of HistogramQueue for recording how the histogram of a dataset has changed over time as new data is added.
    This subclass can be used to recall the updated histogram data after each call to .add()
    """
    def __init__(self, range, bins, maxlen=None, initial_size=100):
        super().__init__(range, bins, maxlen)
        # initial_size is a hint for how large to make the history array (i.e., the expected number of calls to .add())
        # If more calls to .add() are made, the history array will be dynamically resized as necessary
        self._counts_history = np.zeros((initial_size, bins), dtype=int)    # The histogram history
        self._length = 0     # The number of calls to .add() that have been made

    def __len__(self):
        return self._length

    def __iter__(self):
        """Iterating over a HistogramHistory yields the histogram counts after each call to .add() was made."""
        for i in range(self._length):
            yield self._counts_history[i]


    def add(self, new_data):
        """Add new data to the queue."""
        super().add(new_data)

        # Resize our history array if necessary
        if self._length == self._counts_history.shape[0]:
            self._counts_history.resize((self._length * 2, self.bins))   # Double the size

        # Record the new histogram counts
        self._counts_history[self._length, :] = self._counts.copy()
        self._length += 1


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

    print(f"Reading in the replay buffer from: {Path(*replay_dir.parts[-3:])} ...")
    saver = Saver(replay_dir)
    keys_to_keep = ["is_first", "is_last", "is_terminal", "progress"]  # Which steps keys to keep in memory

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
            small_step = {k: step[k] for k in keys_to_keep}
            orig_steps.append(small_step)

            # If this step is the end of an episode, record the step number
            if step["is_last"]:
                episode_steps.append(len(orig_steps))

    print(f"Total number of original (non-augmented) steps in the replay buffer: {len(orig_steps):,}")
    num_episodes = len(episode_steps)
    print(f"Number of complete episodes in the replay buffer: {num_episodes:,}")

    # Validate the data
    print("Validating replay buffer data...")
    if len(all_streams) %4 != 0:
        print("WARNING: Number of streams loaded in was not a multiple of 4!", file=sys.stderr)
    if orig_steps[0]["is_first"] != True:
        print("WARNING: First original step was not is_first=True", file=sys.stderr)
    if orig_steps[-1]["is_last"] != True:
        print("WARNING: Last original step was not is_last=True", file=sys.stderr)
    if orig_steps[-1]["is_terminal"] != True:
        print("WARNING: Last original step was not is_terminal=True", file=sys.stderr)
    num_starts = sum(1 for step in orig_steps if step["is_first"])
    num_lasts = sum(1 for step in orig_steps if step["is_last"])
    if num_starts != num_lasts:
        print(f"WARNING: Partial episodes detected! Number of is_first steps ({num_starts:,}) != number of is_last steps ({num_lasts:,})", file=sys.stderr)

    # Plot the data in the replay buffer
    print("Plotting replay buffer distribution history...")
    assumed_replay_size = int(config.replay_size / 4)   # Only 1/4 of the replay buffer was used for original (non-augmented) data
    print(f" (assuming a replay capacity of {assumed_replay_size:,} non-augmented steps)")

    # NOTE: We don't plot the "terminal" steps in an episode since we never "learn" from them (they have no associated action)
    # Also, the "progress" value of the terminal steps is often very high due to "cheating" that ended the episode
    # We're trying to visualize what parts of the maze DreamerV3 is learning from
    vals = [np.nan if s["is_terminal"] else s["progress"].item() for s in orig_steps]

    # Gather the data for plotting
    replay_history = HistogramHistory(range=(1, 9860), bins=100, maxlen=assumed_replay_size, initial_size=num_episodes)
    # For each episode...
    for start_step, end_step in pairwise([0] + episode_steps):
        # Add the data from this episode and record the new histogram
        replay_history.add(vals[start_step:end_step])

    # 1. Set up the initial plot
    _, _, patches = plt.hist([], bins=replay_history.bin_edges, color="tab:red")
    plt.suptitle("Distribution of \"progress\" Values in the Replay Buffer", fontsize=14)
    title_text = plt.title("")
    plt.xlabel("Progress Along Maze")
    plt.ylabel("% of Replay Buffer Data")
    plt.ylim((0, 0.1))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set_axisbelow(True)
    plt.grid(axis='y', alpha=0.3)

    # 2. Update function
    def update_plot(info):
        # Get the data for this episode
        episode_num, step_num, counts = info

        # Update the rectangle heights
        new_h = counts / counts.sum()    # Express the height as a proportion of all data
        for rect, h in zip(patches, new_h):
            rect.set_height(h)

        # Update the title
        title_text.set_text(f"(after {episode_num:,} episodes  / {step_num:,} steps)")

        return (title_text,) + patches

    # 3. Create the animation
    save_animation = "save_animation" in sys.argv[1:]     # Should we save or display the animation?
    repeat_animation = not save_animation    # Only repeat the animation if we're displaying it
    plot_info = zip(range(1, num_episodes + 1), episode_steps, replay_history)
    ani = FuncAnimation(plt.gcf(), update_plot, frames=plot_info, blit=False, interval=25, repeat=repeat_animation, repeat_delay=5000)

    # 4. Save or display the animation
    if save_animation:
        # Have the user select where to save the video
        save_path = asksaveasfilename(
            title="Save animation video as:",
            initialdir="~/Videos/",
            defaultextension=".mp4",
            filetypes=[("Video files", "*.mp4"), ("All files", "*.*")]
        )

        # If the user didn't cancel...
        if save_path:
            print(f"Saving video to: {save_path}...")
            ani.save(save_path, writer="ffmpeg", fps=30)
            print("Video saved!")

    else:
        # Display the animation instead...
        plt.show()

if __name__ == "__main__":
    main()
