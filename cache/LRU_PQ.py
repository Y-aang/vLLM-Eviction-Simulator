from collections import OrderedDict
import numpy as np
import heapq
import itertools
from tqdm import tqdm

class LRUPQCache:
    def __init__(self, max_size):
        self.max_size = max_size
        # key -> (value, timestamp)
        self._cache = {}
        # min-heap of (timestamp, key)
        self._heap = []
        # 全局单调递增计数器，用作时间戳
        self._counter = itertools.count()
        # 命中统计
        self.hit_count = 0
        self.access_count = 0

    def get(self, key):
        self.access_count += 1
        entry = self._cache.get(key)
        if entry is None:
            return None

        # 命中，更新命中数
        self.hit_count += 1
        value, _ = entry
        # 生成新时间戳并更新字典和堆
        ts = next(self._counter)
        self._cache[key] = (value, ts)
        heapq.heappush(self._heap, (ts, key))
        return value

    def put(self, key, value):
        # 生成新时间戳
        ts = next(self._counter)
        # 无论新增还是更新，都把最新 (value, ts) 放到字典，并入堆
        self._cache[key] = (value, ts)
        heapq.heappush(self._heap, (ts, key))

        # 如果超出容量，就驱逐最旧的那条
        if len(self._cache) > self.max_size:
            # 一直弹堆顶，直到找到一个在字典中时间戳匹配的条目
            while self._heap:
                oldest_ts, oldest_key = heapq.heappop(self._heap)
                # 如果字典里这条记录的时间戳正是 popped 的，就是真正的最旧项
                cur = self._cache.get(oldest_key)
                if cur is not None and cur[1] == oldest_ts:
                    # 驱逐它
                    del self._cache[oldest_key]
                    break

    def hit_rate(self):
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0

def read_block_data_v3(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    data = [[(int(num), str(i + 1)) for num in line.split()]  # 每行作为一个子列表
            for i, line in enumerate(lines) if line]  # 过滤空行
    
    return data

def power_law_sampling(num_elements, sequence_length=10000, exponent=1.0):
    values = np.arange(1, num_elements + 1)
    probabilities = values ** -exponent
    probabilities /= probabilities.sum()
    sampled_indices = np.random.choice(values - 1, size=sequence_length, p=probabilities)
    return [data[i] for i in sampled_indices]


if __name__ == "__main__":
    np.random.seed(42)
    data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/142_docs.txt"
    data = read_block_data_v3(data_path)[:]
    line_lengths = [len(line) for line in data]
    print(line_lengths)
    print("Average length:", np.mean(line_lengths))
    # data = [(1, 'A'), (2, 'B'), (3, 'C'), (1, 'A1'), (4, 'D'), (5, 'E'), (1, 'A2'), (3, 'C1')]
    selected_inputs = power_law_sampling(len(data))
    data = selected_inputs

    cache = LRUPQCache(max_size=600 * 10)     # 2383
    for i, row in enumerate(tqdm(data)):
        for key, value in row:
            # print(f"Accessed ({key}):")
            cache.get(key)
        for key, value in reversed(row):
            cache.put(key, value)
    print(f"Hit Rate: {cache.hit_rate()}")
