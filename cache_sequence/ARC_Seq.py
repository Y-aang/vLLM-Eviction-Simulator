from collections import OrderedDict, deque
import numpy as np
import heapq
from tqdm import tqdm

class ARCSeqCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.T1_heap = []  # (timestamp, key)
        self.T1_data = {}  # key -> (timestamp, value)
        self.T2_heap = []  # (timestamp, key)
        self.T2_data = {}

        self.B1 = deque()
        self.B2 = deque()

        self.p = 0
        self.hit_count = 0
        self.access_count = 0

    def get(self, key):
        self.access_count += 1
        if key in self.T1_data:
            self.hit_count += 1
            return self.T1_data[key][1]
        if key in self.T2_data:
            self.hit_count += 1
            return self.T2_data[key][1]
        return None

    def put(self, key, value, timestamp):
        if key in self.T1_data:
            # Promote to T2
            old_value = self.T1_data.pop(key)[1]
            self.T2_data[key] = (timestamp, old_value)
            heapq.heappush(self.T2_heap, (timestamp, key))
            return

        if key in self.T2_data:
            # Refresh T2
            self.T2_data[key] = (timestamp, value)
            heapq.heappush(self.T2_heap, (timestamp, key))
            return

        if key in self.B1:
            # print("hit B1")
            delta = max(1, len(self.B2) // max(1, len(self.B1)))
            self.p = min(self.p + delta, self.max_size)
            self.B1.remove(key)
            if len(self.T1_data) + len(self.T2_data) >= self.max_size:
                self._replace(key)
            self.T2_data[key] = (timestamp, value)
            heapq.heappush(self.T2_heap, (timestamp, key))
            return

        if key in self.B2:
            delta = max(1, len(self.B1) // max(1, len(self.B2)))
            self.p = max(self.p - delta, 0)
            self.B2.remove(key)
            if len(self.T1_data) + len(self.T2_data) >= self.max_size:
                self._replace(key)
            self.T2_data[key] = (timestamp, value)
            heapq.heappush(self.T2_heap, (timestamp, key))
            return

        # 新key插入，严格遵循原始逻辑
        # print("miss")
        L1_size = len(self.T1_data) + len(self.B1)
        if L1_size == self.max_size:
            if len(self.T1_data) < self.max_size:
                if self.B1:
                    self.B1.popleft()
                    self._replace(key)
            else:
                if self.T1_data:
                    self._evict_from_T1()
        elif L1_size < self.max_size:
            total_size = len(self.T1_data) + len(self.T2_data) + len(self.B1) + len(self.B2)
            if total_size >= self.max_size:
                if total_size == 2 * self.max_size and self.B2:
                    self.B2.popleft()
                self._replace(key)

        # 使用外部传入的时间戳
        self.T1_data[key] = (timestamp, value)
        heapq.heappush(self.T1_heap, (timestamp, key))
        self._prune_ghosts()

    def _replace(self, key):
        if self.T1_data and ((key in self.B2 and len(self.T1_data) == self.p) or (len(self.T1_data) > self.p)):
            self._evict_from_T1()
        elif self.T2_data:
            self._evict_from_T2()

    def _evict_from_T1(self):
        while self.T1_heap:
            timestamp, key = heapq.heappop(self.T1_heap)
            if key in self.T1_data and self.T1_data[key][0] == timestamp:
                self.B1.append(key)
                del self.T1_data[key]
                return

    def _evict_from_T2(self):
        while self.T2_heap:
            timestamp, key = heapq.heappop(self.T2_heap)
            if key in self.T2_data and self.T2_data[key][0] == timestamp:
                self.B2.append(key)
                del self.T2_data[key]
                return

    def _prune_ghosts(self):
        while len(self.B1) > self.max_size:
            assert False
            self.B1.popleft()
        while len(self.B2) > self.max_size:
            assert False
            self.B2.popleft()

    def hit_rate(self):
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0


def read_block_data_v3(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    data = [[(int(num), str(i + 1)) for num in line.split()]
            for i, line in enumerate(lines) if line]
    return data

def power_law_sampling(data, num_elements, sequence_length=1000, exponent=1.0):
    values = np.arange(1, num_elements + 1)
    probabilities = values ** -exponent
    probabilities /= probabilities.sum()
    sampled_indices = np.random.choice(values - 1, size=sequence_length, p=probabilities)
    return [data[i] for i in sampled_indices]


if __name__ == "__main__":
    np.random.seed(42)
    data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/142_docs.txt"
    data = read_block_data_v3(data_path)
    data = power_law_sampling(data, len(data))

    cache = ARCSeqCache(max_size=600 * 10)

    for seq_id, row in enumerate(tqdm(data)):
        for word_id, (key, value) in enumerate(row):
            timestamp = (seq_id, -word_id)  # 外部传入的时间戳
            cache.get(key)
        for word_id, (key, value) in enumerate(row):
            timestamp = (seq_id, -word_id)  # 外部传入的时间戳
            cache.put(key, value, timestamp)
        # print(f"Step {seq_id+1} ARCSeqCache T1: {len(cache.T1_data)}, T2: {len(cache.T2_data)}, B1: {len(cache.B1)}, B2: {len(cache.B2)}, p: {cache.p}")
        # print(f"Hit Rate: {cache.hit_rate():.2%}")
        
    print(f"Hit Rate: {cache.hit_rate():.2%}")
