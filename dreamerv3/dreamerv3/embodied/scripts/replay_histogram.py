import sys
from collections import deque
from itertools import pairwise, chain
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tkinter import Tk
from tkinter.filedialog import askdirectory, asksaveasfilename
from dreamerv3.embodied import Config
from dreamerv3.embodied.replay.saver import Saver
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from matplotlib.animation import FuncAnimation

class HistogramQueue:
    """
    A class for efficiently maintaining histogram information for a collection of values
    as new values are added to the collection. Rather than re-calculating the entire
    histogram after new values are added, the counts for the new values are added to existing counts.

    There are two separate ways to have the histogram information reflect only the most "recent" data:
      1. Define a value for maxlen, which defines an upper limit on how much (recent) data is used to calculate the histogram values
      2. Define a value for decay, which defines the proportion of the previous counts that are maintained before new counts are added
    Only one of theese two options may be used at a time. maxlen is best for fixed-sized queues in which old is completely discarded.
    decay is best when the influence of old values decreases over time.
    """
    def __init__(self, range, bins, maxlen=None, decay=None):
        self.range = range     # The lower and upper range of the histogram bins
        self.bins = bins       # The number of bins in the histogram
        self.maxlen = maxlen   # Only the most recent maxlen data items will be used for the histogram values
        self.decay = decay     # Previous histogram counts will be multiplied by this value before new counts are added

        # Make sure that maxlen and decay are used exclusively
        assert not (maxlen and decay), "You cannot use both maxlen and decay at the same time."

        self.bin_edges = np.histogram_bin_edges([], bins=bins, range=range)
        self._counts = np.zeros(bins, dtype="float" if decay else "int")
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

            # We can now add this data without overflowing
            self._data.extend(new_data)
            data_counts, _ = np.histogram(new_data, bins=self.bin_edges)
            if self.decay:
                self._counts *= self.decay
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
    def __init__(self, range, bins, maxlen=None, decay=None, initial_size=100):
        super().__init__(range, bins, maxlen, decay)
        # initial_size is a hint for how large to make the history array (i.e., the expected number of calls to .add())
        # If more calls to .add() are made, the history array will be dynamically resized as necessary
        self._counts_history = np.zeros(shape=(initial_size, bins), dtype=self._counts.dtype)    # The histogram history
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
    if not orig_steps[0]["is_first"]:
        print("WARNING: First original step was not is_first=True", file=sys.stderr)
    if not orig_steps[-1]["is_last"]:
        print("WARNING: Last original step was not is_last=True", file=sys.stderr)
    if not orig_steps[-1]["is_terminal"]:
        print("WARNING: Last original step was not is_terminal=True", file=sys.stderr)
    num_starts = sum(1 for step in orig_steps if step["is_first"])
    num_lasts = sum(1 for step in orig_steps if step["is_last"])
    if num_starts != num_lasts:
        print(f"WARNING: Partial episodes detected! Number of is_first steps ({num_starts:,}) != number of is_last steps ({num_lasts:,})", file=sys.stderr)

    # Gather the data for plotting
    print("Calculating distribution histories...")
    assumed_replay_size = int(config.replay_size / 4)   # Only 1/4 of the replay buffer was used for original (non-augmented) data
    print(f" (assuming a replay buffer capacity of {assumed_replay_size:,} non-augmented steps)")

    # NOTE: We don't plot the "terminal" steps in an episode since we never "learn" from them (they have no associated action)
    # Also, the "progress" value of the terminal steps is often very high due to "cheating" that ended the episode
    # We're trying to visualize what parts of the maze DreamerV3 is learning from
    progress_vals = [np.nan if s["is_terminal"] else s["progress"].item() for s in orig_steps]

    hist_range = (1, 9860)    # Range of "progress" values for CyberRunner
    num_bins = 100            # Number of bins in the histograms
    sampling = "uniform"      # or "priority"
    replay_history = HistogramHistory(range=hist_range, bins=num_bins, maxlen=assumed_replay_size, initial_size=num_episodes)
    training_history = HistogramHistory(range=hist_range, bins=num_bins, maxlen=1_000_000, initial_size=num_episodes)    # maxlen defines what is "recent" for the training data
    rng = np.random.default_rng(seed=0)
    seq_len = config.batch_length
    training_ratio = config.run.train_ratio / (config.batch_size * config.batch_length)

    # For each episode...
    for start_step, end_step in pairwise([0] + episode_steps):
        # Add the data from this episode to our replay buffer history and record the new histogram
        replay_history.add(progress_vals[start_step:end_step])

        # Sample some training data from the replay buffer
        replay_contents = np.array(replay_history._data)
        ep_len = end_step - start_step
        num_samples = int(ep_len * training_ratio)

        match sampling:
            case "uniform":
                # Uniform sampling
                seq_idx = rng.integers(len(replay_contents) - seq_len + 1, size=num_samples)

            case "priority":
                # Prioritized sampling
                window = sliding_window_view(replay_contents, window_shape=seq_len)
                seq_maxes = np.nanmax(window, axis=1)
                probs = (0.4 + seq_maxes / np.nanmax(replay_contents)) ** 0.8
                seq_idx = rng.choice(len(probs), size=num_samples, p=probs / probs.sum())

            case _:
                # Unsupported sampling type
                raise ValueError(f"Sampling type '{sampling}' is not supported")

        # Add all the sampled steps to the training_history and record the new histogram
        sampled_sequences = [replay_contents[idx:idx + seq_len] for idx in seq_idx]
        training_history.add(list(chain.from_iterable(sampled_sequences)))

    # Animate the distribution histories
    print("Plotting the distribution histories...")

    # 1. Set up the initial plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex='all', sharey='all')
    replay_hist, training_hist = axs
    title_text = plt.suptitle("", fontsize=14)

    # Distribution of data in the replay buffer
    _, _, rb_patches = replay_hist.hist([], bins=replay_history.bin_edges, color="tab:red")
    replay_hist.set_title("Replay Buffer")
    replay_hist.set_xlabel("Progress Along Maze")
    replay_hist.set_ylabel("% of Replay Buffer Data")
    replay_hist.set_ylim((0, 0.1))
    replay_hist.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    replay_hist.yaxis.set_major_formatter(PercentFormatter(1))
    replay_hist.set_axisbelow(True)
    replay_hist.grid(axis='y', alpha=0.3)

    # Distribution of training data
    _, _, td_patches = training_hist.hist([], bins=training_history.bin_edges, color="tab:blue")
    training_hist.set_title("Training Data")
    training_hist.set_xlabel("Progress Along Maze")
    training_hist.set_ylabel("% of Training Data")
    training_hist.set_axisbelow(True)
    training_hist.grid(axis='y', alpha=0.3)

    # 2. Update function
    def update_plot(info):
        # Get the data for this episode
        episode_num, step_num, rb_counts, td_counts = info

        # Update the rectangle heights
        new_h = rb_counts / rb_counts.sum()    # Express the height as a proportion of all data
        for rect, h in zip(rb_patches, new_h):
            rect.set_height(h)

        new_h = td_counts / td_counts.sum()    # Express the height as a proportion of all data
        for rect, h in zip(td_patches, new_h):
            rect.set_height(h)

        # Update the title
        title_text.set_text(f"Distributions after {episode_num:,} episodes  / {step_num:,} steps")

        return (title_text,) + rb_patches + td_patches

    # 3. Create the animation
    save_animation = "save_animation" in sys.argv[1:]     # Should we save or display the animation?
    repeat_animation = not save_animation    # Only repeat the animation if we're displaying it
    plot_info = zip(range(1, num_episodes + 1), episode_steps, replay_history, training_history)
    ani = FuncAnimation(fig, update_plot, frames=plot_info, interval=0, repeat=repeat_animation, repeat_delay=5000)

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
