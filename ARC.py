from collections import OrderedDict, deque
import numpy as np

class ARCCache:
    def __init__(self, max_size):
        self.max_size = max_size  # 缓存容量 c
        # T1 和 T2 存储实际数据
        self.T1 = OrderedDict()  # 最近访问但访问次数不多的项（短期 LRU）
        self.T2 = OrderedDict()  # 频繁访问的项（长期 LFU）
        # B1 和 B2 为 ghost 列表，只记录被淘汰项的 key
        self.B1 = deque()
        self.B2 = deque()
        # 动态平衡参数 p
        self.p = 0
        
        # 命中和访问统计
        self.hit_count = 0
        self.access_count = 0

    def get(self, key):
        """
        仅检查 T1 和 T2：
        - 如果命中，则记录命中数和访问数，返回缓存的 value；
        - 不做任何更新或提升操作。
        """
        self.access_count += 1
        if key in self.T1:
            self.hit_count += 1
            return self.T1[key]
        if key in self.T2:
            self.hit_count += 1
            return self.T2[key]
        return None

    def put(self, key, value):
        """
        写入接口：
        - 如果 key 已存在于 T1/T2，则更新 value（可选择不做额外调整，因为 get 不修改状态）；
        - 如果 key 在 ghost 列表 B1 或 B2 中，根据 ARC 算法先调整参数 p，
          并将该 key 从 ghost 列表中移除，随后插入 T2；
        - 如果 key 全新：
            * 当缓存未满时，将 key 插入 T1；
            * 当缓存已满时，先调用 _replace(key) 淘汰数据，再将 key 插入 T1；
        - 最后确保 ghost 列表大小不超过 max_size。
        """
        # 若 key 已存在于实际数据中，直接更新 value
        if key in self.T1:
            # 提升到 T2 的 MRU（同时更新 value）
            self.T2[key] = self.T1.pop(key)
            self.T2.move_to_end(key, last=True)
            return
        elif key in self.T2:
            # 更新 value，并移动到 MRU
            self.T2[key] = value
            self.T2.move_to_end(key, last=True)
            return

        # 如果 key 在 ghost 列表中
        if key in self.B1:
            # 根据 ARC 算法调整 p
            delta = max(1, len(self.B2) // max(1, len(self.B1)))
            self.p = min(self.p + delta, self.max_size)
            self.B1.remove(key)
            if len(self.T1) + len(self.T2) >= self.max_size:
                self._replace(key)
            self.T2[key] = value
            return

        if key in self.B2:
            delta = max(1, len(self.B1) // max(1, len(self.B2)))
            self.p = max(self.p - delta, 0)
            self.B2.remove(key)
            if len(self.T1) + len(self.T2) >= self.max_size:
                self._replace(key)
            self.T2[key] = value
            return

        # 新 key  有点不一样
        # if len(self.T1) + len(self.T2) < self.max_size:
        #     self.T1[key] = value
        # else:
        #     self._replace(key)
        #     self.T1[key] = value
        # self._prune_ghosts()
        
        # 新 key 插入：严格按照 ARC 伪代码的 Case IV 实现
        L1_size = len(self.T1) + len(self.B1)
        if L1_size == self.max_size:
            if len(self.T1) < self.max_size:
                if self.B1:
                    self.B1.popleft()  # 删除 B1 的 LRU
            else:
                if self.T1:
                    self.T1.popitem(last=False)  # 删除 T1 的 LRU
            self._replace(key)
        elif L1_size < self.max_size:
            total_size = len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2)
            if total_size >= self.max_size:
                if total_size == 2 * self.max_size and self.B2:
                    self.B2.popleft()  # 删除 B2 的 LRU
                self._replace(key)
        # 插入新 key 到 T1 的 MRU 位置
        self.T1[key] = value
        self._prune_ghosts()

    def _replace(self, key):
        """
        REPLACE 子过程：
        - 如果 T1 非空且满足：
             ( (key 在 B2 中 且 |T1| == p) 或 (|T1| > p) )
          则从 T1 淘汰最旧的项，并将其 key 放入 B1；
        - 否则，从 T2 淘汰最旧的项，并将其 key 放入 B2。
        """
        if self.T1 and ((key in self.B2 and len(self.T1) == self.p) or (len(self.T1) > self.p)):
            old_key, _ = self.T1.popitem(last=False)
            self.B1.append(old_key)
        elif self.T2:
            old_key, _ = self.T2.popitem(last=False)
            self.B2.append(old_key)

    def _prune_ghosts(self):
        """ 保证 ghost 列表 B1 和 B2 的大小不超过 max_size """
        while len(self.B1) > self.max_size:
            self.B1.popleft()
        while len(self.B2) > self.max_size:
            self.B2.popleft()

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
    data = read_block_data_v3(data_path)
    line_lengths = [len(line) for line in data]
    print(line_lengths)
    print("Average length:", np.mean(line_lengths))

    data = power_law_sampling(len(data))

    cache = ARCCache(max_size=600 * 10)  # 或 LRUCache
    for i, row in enumerate(data):
        print('i:', i)
        for key, value in row:
            cache.get(key)
        for key, value in reversed(row):
            cache.put(key, value)

    print(f"Hit Rate: {cache.hit_rate():.2%}")
