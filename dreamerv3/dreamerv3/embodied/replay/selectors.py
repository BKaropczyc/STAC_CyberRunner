from collections import deque, defaultdict
from embodied.core.histogram import DecayingHistogramStream
from itertools import chain
import threading

import numpy as np

class SequentialSet:
    """A simple set-like object that can return its elements as a sequence in O(1) time."""
    def __init__(self):
        self.items = []
        self.indices = {}

    def add(self, item):
        assert item not in self.indices, "SequentialSet items must be unique!"
        self.indices[item] = len(self.items)
        self.items.append(item)

    def remove(self, item):
        index = self.indices.pop(item)
        last = self.items.pop()
        if index != len(self.items):
            self.items[index] = last
            self.indices[last] = index

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def index(self, item):
        return self.indices[item]

    def as_sequence(self):
        return self.items


class Fifo:
    def __init__(self):
        self.queue = deque()

    def __call__(self):
        return self.queue[0]

    def __setitem__(self, key, steps):
        self.queue.append(key)

    def __delitem__(self, key):
        if self.queue[0] == key:
            self.queue.popleft()
        else:
            # TODO: This branch is unused but very slow.
            self.queue.remove(key)


class Uniform:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.keys = SequentialSet()

    def __call__(self):
        index = self.rng.integers(len(self.keys))
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.keys.add(key)

    def __delitem__(self, key):
        self.keys.remove(key)


class Prio:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.sampled_indices = deque()
        self.sample_lock = threading.Lock()
        self.keys = SequentialSet()
        self.prios = []
        self.max_progress = 0

    def __call__(self):
        # index = self.rng.integers(0, len(self.keys)).item()
        # Make sure we populate sampled_indices atomically
        with self.sample_lock:
            if not self.sampled_indices:  # TODO this is wrong when replay buffer is full
                probs = (0.4 + np.asarray(self.prios) / self.max_progress) ** 0.8
                self.sampled_indices.extend(self.rng.choice(len(probs), 512, p=probs / probs.sum()))
        # probs = (0.4 + np.asarray(self.prios) / self.max_progress) ** 0.8
        # index = self.rng.choice(len(self.keys), p=probs/probs.sum())
        index = self.sampled_indices.pop()
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.keys.add(key)
        progress = np.asarray([step["progress"] for step in steps])
        max_progress = progress.max()
        self.max_progress = max(self.max_progress, max_progress)
        self.prios.append(max_progress)

    def __delitem__(self, key):
        index = self.keys.index(key)
        self.keys.remove(key)
        last_prio = self.prios.pop()
        if index != len(self.prios):
            self.prios[index] = last_prio


class UniformTraining:
    """
    A selector which biases its sampling to return a roughly uniform distribution of samples according to their progress.
    This attempts to train all parts of the maze equally, and rapidly trains new sections of the maze as they are reached.
    """
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

        # We perform the sampling of sequence keys in bulk, to save time.
        self.sampled_keys = deque()
        self.sample_lock = threading.Lock()
        self.max_prefetch = None    # Limit how many samples are pre-fetched
        self.training_histogram = DecayingHistogramStream(bins=100, range=(1, 9860), decay_batch=10_000, decay_rate=0.99)

        self.key_to_bin = {}         # Dict of sequence key -> histogram bin the sequence is considered to be in (based on mean progress)
        self.bin_to_keys = defaultdict(SequentialSet)      # Dict of histogram bin -> list of keys added to the bin
        self.key_to_progress = {}    # Dict of sequence key -> progress values array for all sequence steps

    def __call__(self):
        # Make sure we populate sampled_keys atomically
        with self.sample_lock:
            # If we need to sample more keys...
            if not self.sampled_keys:
                # Determine how many samples we should pre-fetch
                sample_size = 512   # Default sample size
                if self.max_prefetch is None:
                    # We haven't completed a training episode yet, so this is for initialization
                    sample_size = 24
                elif self.max_prefetch <= 0:
                    print(f"WARNING: The UniformTraining sampler is being asked to pre-fetch more samples than expected!")
                    print(f"   max_prefetch is currently {self.max_prefetch:,}")
                else:
                    # Make sure we don't pre-fetch more samples than needed
                    sample_size = min(sample_size, self.max_prefetch)
                    self.max_prefetch -= sample_size    # Decrement max_prefetch for next time

                if self.training_histogram.counts().sum() == 0:
                    # We haven't sampled any training data yet.
                    # Just sample uniformly from the available sequence keys.
                    all_keys = list(self.key_to_bin.keys())
                    sampled_idx = self.rng.integers(len(all_keys), size=sample_size)
                    self.sampled_keys.extend(all_keys[i] for i in sampled_idx)

                else:
                    # Bias the sampling to attempt to "even out" the amount of training data along the entire maze path

                    # For each bin we have experience data for, decide what proportion of the prior training data came from that bin
                    bins_with_data = [k for k, v in self.bin_to_keys.items() if len(v) > 0]
                    bin_counts = self.training_histogram.counts()[bins_with_data]
                    bin_proportions = bin_counts / bin_counts.sum()

                    # Invert these proportions, so that bins with little existing training data get higher values,
                    # and bins with lots of existing training data get lower values.
                    # The 0.1 "smooths" the values a little, so that even the bin with the most training data has a >0 probability of being sampled
                    # Small values (as low as 0.0) encourage the sampling to resemble the inverse training data distribution
                    # Larger values yield a more uniform sampling from among the bins that contain experience data
                    inverse_proportions = bin_proportions.max() + 0.1 - bin_proportions

                    # Select bins to sample from according to these weights.
                    # Bins with lots of training data have a lower probability of being selected and vice versa.
                    # Sampling bins rather than experience sequences isolates the sample from the biased distribution of the replay buffer
                    bin_selections = self.rng.choice(bins_with_data, size=sample_size, p=inverse_proportions / inverse_proportions.sum())

                    # Randomly choose an experience sequence from each selected bin
                    self.sampled_keys.extend((bin_keys:=self.bin_to_keys[bin])[self.rng.integers(len(bin_keys))] for bin in bin_selections)

                # If this request is in response to an actual training episode...
                if self.max_prefetch is not None:
                    # Add the progress values of these sequences to our training data histogram
                    progress_of_samples = [self.key_to_progress[k] for k in self.sampled_keys]
                    self.training_histogram.add(list(chain.from_iterable(progress_of_samples)))

        # Return the key for a single sampled sequence
        return self.sampled_keys.pop()

    def __setitem__(self, key, steps):
        # Decide which histogram bin the experience data sequence should be considered in when sampling
        progress_vals = np.array([s["progress"].item() for s in steps if not s["is_terminal"]], dtype="uint16")   # Don't include terminal steps, which could be cheating
        mean_progress = np.mean(progress_vals)     # Use the average progress value to decide the bin
        bin = np.digitize(mean_progress, bins=self.training_histogram.bin_edges[:-1]) - 1  # digitize() returns len(bins) for values above the last entry in bins

        # Add this key to our data structures
        self.key_to_bin[key] = bin
        self.bin_to_keys[bin].add(key)
        self.key_to_progress[key] = progress_vals

    def __delitem__(self, key):
        # Remove this key from our data structures
        del self.key_to_progress[key]
        self.bin_to_keys[self.key_to_bin[key]].remove(key)
        del self.key_to_bin[key]

    def set_max_prefetch(self, max_prefetch):
        """
        Set the maximum number of samples that should be pre-fetched to avoid having samples left over at the end of an episode.
        When the replay buffer is full, pre-fetching excess samples can result in the samples no longer being valid next episode
        (e.g., if they are expelled from the replay buffer as new data is added).
        """
        if len(self.sampled_keys) > max_prefetch:
            # We've already pre-fetched too much!
            # This can happen during training initialization
            print(f"WARNING: Removing {len(self.sampled_keys) - max_prefetch:,} pre-fetched samples from the UniformTraining sampler.")
            while len(self.sampled_keys) > max_prefetch:
                self.sampled_keys.pop()

        # If we've already pre-fetched samples, decrease the limit by that amount
        max_prefetch -= len(self.sampled_keys)

        # Remember how many samples we can pre-fetch in the future
        self.max_prefetch = max_prefetch
