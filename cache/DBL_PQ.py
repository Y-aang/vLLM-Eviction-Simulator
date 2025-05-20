from collections import OrderedDict, deque
import numpy as np
import heapq
import itertools
import math

class DBLCachePQ:
    def __init__(self, max_size):
        self.k = int(max_size * 0.5)
        self.max_size = math.ceil(max_size)
        
        self.A1in_heap = []     # heap of (timestamp, key)
        self.A1in_data = {}     # key -> (timestamp, value)
        
        self.Am_heap = []       # heap of (timestamp, key)
        self.Am_data = {}       # key -> (timestamp, value)

        self.time = itertools.count()  # global timestamp generator

        self.hit_count = 0
        self.access_count = 0

    def get(self, key):
        self.access_count += 1

        if key in self.Am_data:
            self.hit_count += 1
            value = self.Am_data[key][1]
            return value

        if key in self.A1in_data:
            self.hit_count += 1
            value = self.A1in_data[key][1]
            return value

        return None

    def put(self, key, value):
        if key in self.Am_data:
            timestamp = next(self.time)
            self.Am_data[key] = (timestamp, value)
            heapq.heappush(self.Am_heap, (timestamp, key))
            return

        if key in self.A1in_data:
            value = self.A1in_data.pop(key)[1]
            timestamp = next(self.time)
            self.Am_data[key] = (timestamp, value)
            heapq.heappush(self.Am_heap, (timestamp, key))
            return

        # insert new key
        assert len(self.A1in_data) <= self.k
        if len(self.A1in_data) == self.k:
            self._evict_from_A1in()

        assert len(self.A1in_data) + len(self.Am_data) <= self.max_size
        if len(self.A1in_data) + len(self.Am_data) == self.max_size:
            self._evict_from_Am()

        timestamp = next(self.time)
        self.A1in_data[key] = (timestamp, value)
        heapq.heappush(self.A1in_heap, (timestamp, key))

    def _evict_from_A1in(self):
        while self.A1in_heap:
            timestamp, key = heapq.heappop(self.A1in_heap)
            if key in self.A1in_data and self.A1in_data[key][0] == timestamp:
                del self.A1in_data[key]
                return

    def _evict_from_Am(self):
        while self.Am_heap:
            timestamp, key = heapq.heappop(self.Am_heap)
            if key in self.Am_data and self.Am_data[key][0] == timestamp:
                del self.Am_data[key]
                return

    def hit_rate(self):
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0

def read_block_data_v3(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    data = [[(int(num), str(i + 1)) for num in line.split()]
            for i, line in enumerate(lines) if line]
    
    return data

def power_law_sampling(num_elements, sequence_length=1000, exponent=1.0):
    values = np.arange(1, num_elements + 1)
    probabilities = values ** -exponent
    probabilities /= probabilities.sum()
    sampled_indices = np.random.choice(values - 1, size=sequence_length, p=probabilities)
    return [data[i] for i in sampled_indices]

if __name__ == "__main__":
    np.random.seed(42)
    data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/142_docs.txt"
    data = read_block_data_v3(data_path)[:]
    # data = [(1, 'A'), (2, 'B'), (3, 'C'), (1, 'A1'), (4, 'D'), (5, 'E'), (1, 'A2'), (3, 'C1')]
    selected_inputs = power_law_sampling(len(data))
    data = selected_inputs

    cache = DBLCachePQ(max_size=600 * 10)     # 2383
    for i, row in enumerate(data):
        for key, value in row:
            # print(f"Accessed ({key}):")
            cache.get(key)
        for key, value in reversed(row):
            cache.put(key, value)
        
        # print(f"{i} round - Hit Rate: {cache.hit_rate()} Hit count: {cache.hit_count} Access count: {cache.access_count}")

    # print("\nFinal Cache State:")
    # print("A1in:", cache.A1in)
    # print("A1out:", list(cache.A1out))
    # print("Am:", cache.Am)
    print(f"Hit Rate: {cache.hit_rate():.2%}")
