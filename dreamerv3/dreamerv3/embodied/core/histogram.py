from collections import deque
import numpy as np


class HistogramStream:
    """
    A class for efficiently maintaining histogram information for a stream of numeric data.
    """
    def __init__(self, bins:int, range:tuple[int | float, int | float]):
        self.bins = bins       # The number of equal-width bins in the histogram
        self.range = range     # The lower and upper range (inclusive) of the histogram bins
        self.size = 0          # The total number of values that have been added to the stream

        self.bin_edges = np.histogram_bin_edges([], bins=bins, range=range)
        self._counts = np.zeros(bins, dtype=np.int64)

    def add(self, new_data):
        """Add new data to the stream."""
        # Update our size
        self.size += len(new_data)

        # Update our counts
        data_counts, _ = np.histogram(new_data, bins=self.bin_edges)
        self._counts += data_counts

    def counts(self, as_proportion:bool=False):
        """Return the current histogram counts, optionally as a proportion of all the (recent) data."""
        if as_proportion:
            return (self._counts / np.sum(self._counts)).copy()
        else:
            return self._counts.copy()


class FixedLenHistogramStream(HistogramStream):
    """
    A HistogramStream subclass for which the histogram counts reflect only a fixed amount of the most recent data added.
    This is best for modeling fixed-size streams in which the oldest data is completely ignored.
    """
    def __init__(self, bins:int, range:tuple[int | float, int | float], maxlen:int):
        super().__init__(bins, range)
        assert maxlen > 0, "maxlen must be positive."
        self.maxlen = maxlen   # The histogram counts will only reflect the most recent maxlen data values
        self._data = deque(maxlen=maxlen)   # A FIFO queue of only "recent" data (defined by maxlen)

    def add(self, new_data):
        """Add new data to the stream."""
        # Update our size
        new_data_len = len(new_data)
        self.size += new_data_len

        if new_data_len >= self.maxlen:
            # The new data will completely replace all existing data
            if new_data_len > self.maxlen:
                # Trim the new data to only the most recent maxlen values
                new_data = new_data[-self.maxlen:]

            self._data = deque(new_data, maxlen=self.maxlen)
            self._counts, _ = np.histogram(new_data, bins=self.bin_edges)

        else:
            # At least some of the existing data will remain
            num_overflow = len(self._data) + new_data_len - self.maxlen
            if num_overflow > 0:
                # We will overflow our maxlen by adding the new data
                # Remove the data that will overflow
                removed_items = [self._data.popleft() for _ in range(num_overflow)]
                remove_counts, _ = np.histogram(removed_items, bins=self.bin_edges)
                self._counts -= remove_counts

            # We can now add this data without overflowing
            self._data.extend(new_data)

            # Update our counts
            data_counts, _ = np.histogram(new_data, bins=self.bin_edges)
            self._counts += data_counts


class DecayingHistogramStream(HistogramStream):
    """
    A HistogramStream subclass for which stream data has a smaller weight on the histogram counts the older it is.
    Histogram counts are no longer integers, but rather floating point values that represent the decaying weight of old data.
    This is best for modeling streams in which the influence of old data decreases over time but is never fully "forgotten".
    """
    def __init__(self, bins:int, range:tuple[int | float, int | float], decay_batch:int, decay_rate:float):
        super().__init__(bins, range)
        self._counts = np.zeros(bins, dtype=np.float64)   # Counts are no longer integers
        assert decay_batch > 0, "decay_batch must be positive."
        self.decay_batch = decay_batch    # The histogram counts will be "decayed" each time this many values are added
        assert 0.0 <= decay_rate <= 1.0, "decay_rate should be a value in [0, 1]"
        self.decay_rate = decay_rate      # Previous histogram counts will be "decayed" by multiplying by this factor

    def add(self, new_data):
        """Add new data to the stream."""
        # While we have data to add...
        while len(new_data) > 0:
            # Decay the counts if we've added a full batch
            if self.size % self.decay_batch == 0:
                self._counts *= self.decay_rate

            # Grab at most decay_batch new data
            batch_len = self.decay_batch - (self.size % self.decay_batch)
            next_batch = new_data[:batch_len]
            new_data = new_data[batch_len:]

            # Update the size and counts
            self.size += len(next_batch)
            data_counts, _ = np.histogram(next_batch, bins=self.bin_edges)
            self._counts += data_counts


class HistogramStreamHistory:
    """
    A class for recording how the histogram of a numeric dataset has changed over time as new data is added.
    A snapshot of the data's distribution is taken upon every call to .add()
    """
    def __init__(self, hist_stream:HistogramStream, initial_size:int=100):
        self.hist_stream = hist_stream   # A preconfigured HistogramStream or subclass
        # initial_size is a hint for how large to make the history array (i.e., the expected number of calls to .add())
        # If more calls to .add() are made, the history array will be dynamically resized as necessary
        self._counts_history = np.zeros(shape=(initial_size, hist_stream.bins), dtype=hist_stream.counts().dtype)    # The histogram history
        self._length = 0     # The current length of the history (i.e., number of calls to .add() that have been made)

    def __len__(self):
        return self._length

    def __iter__(self):
        """Iterating over a HistogramStreamHistory yields the histogram counts after each call to .add() was made."""
        for i in range(self._length):
            yield self._counts_history[i]

    def add(self, new_data):
        """Add the new data to the histogram stream."""
        self.hist_stream.add(new_data)

        # Resize our history array if necessary
        if self._counts_history.shape[0] == self._length:
            self._counts_history.resize((self._length * 2, self.hist_stream.bins))   # Double the size

        # Record the new histogram counts
        self._counts_history[self._length, :] = self.hist_stream.counts()
        self._length += 1
