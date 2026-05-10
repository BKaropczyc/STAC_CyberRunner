import concurrent.futures
from collections import defaultdict, deque
from functools import partial as bind

import embodied

from . import chunk as chunklib


class Saver:
    def __init__(self, directory, chunks=1024):
        self.directory = embodied.Path(directory)
        self.directory.mkdirs()
        self.chunks = chunks
        self.buffers = defaultdict(bind(chunklib.Chunk, chunks))
        self.workers = concurrent.futures.ThreadPoolExecutor(16)
        self.promises = deque()
        self.loading = False

    def add(self, step, worker):
        if self.loading:
            return
        buffer = self.buffers[worker]
        buffer.append(step)
        if buffer.length >= self.chunks:
            print("Saving chunk")
            self.buffers[worker] = buffer.successor = chunklib.Chunk(self.chunks)
            self.promises.append(self.workers.submit(buffer.save, self.directory))
            for promise in [x for x in self.promises if x.done()]:
                promise.result()
                self.promises.remove(promise)

    def save(self, wait=False):
        for buffer in self.buffers.values():
            if buffer.length:
                self.promises.append(self.workers.submit(buffer.save, self.directory))
        if wait:
            [x.result() for x in self.promises]
            self.promises.clear()

    def load(self, capacity, length, low_memory=False):
        filenames = chunklib.Chunk.scan(self.directory, capacity, length - 1)
        if not filenames:
            return
        streamids = {}
        for filename in reversed(filenames):
            timestamp, uuid, successor, length = filename.stem.split("-")
            if successor not in streamids:
                streamids[uuid] = int(embodied.uuid())
            else:
                streamids[uuid] = streamids[successor]
        self.loading = True
        if low_memory == True:
            # Conserve memory by reading in the chucks sequentially
            for filename in filenames:
                chunk = chunklib.Chunk.load(filename)
                stream = streamids[chunk.uuid]
                for index in range(chunk.length):
                    step = {k: v[index] for k, v in chunk.data.items()}
                    yield step, stream
        else:
            threads = min(len(filenames), 32)
            with concurrent.futures.ThreadPoolExecutor(threads) as executor:
                chunks = list(executor.map(chunklib.Chunk.load, filenames))
            for i, chunk in enumerate(chunks):
                stream = streamids[chunk.uuid]
                for index in range(chunk.length):
                    step = {k: v[index] for k, v in chunk.data.items()}
                    yield step, stream
                # Free memory early to not require twice the replay capacity.
                chunks[i] = None
                del chunk
        self.loading = False
