from collections import OrderedDict
import numpy as np

flag = False

class LRUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hit_count = 0
        self.access_count = 0
    
    def get(self, key):
        self.access_count += 1

        if key in self.cache:
            self.hit_count += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        global flag
        self.cache[key] = value
        self.cache.move_to_end(key)

        if len(self.cache) > self.max_size:
            # print("evict")
            flag = True
            self.cache.popitem(last=False)

    def hit_rate(self):
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0


def read_block_data_v3(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    data = [[(int(num), str(i + 1)) for num in line.split()]  # 每行作为一个子列表
            for i, line in enumerate(lines) if line]  # 过滤空行
    
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
    line_lengths = [len(line) for line in data]
    print(line_lengths)
    print("Average length:", np.mean(line_lengths))
    # data = [(1, 'A'), (2, 'B'), (3, 'C'), (1, 'A1'), (4, 'D'), (5, 'E'), (1, 'A2'), (3, 'C1')]
    selected_inputs = power_law_sampling(len(data))
    data = selected_inputs

    cache = LRUCache(max_size=600 * 10)     # 2383
    for i, row in enumerate(data):
        for key, value in row:
            # print(f"Accessed ({key}):")
            cache.get(key)
        for key, value in reversed(row):
            cache.put(key, value)
        
        # if flag==True:
        #     print('evict happen')
        # print(f"{i} round - Hit Rate: {cache.hit_rate()} Hit count: {cache.hit_count} Access count: {cache.access_count}")

    # # print("\nFinal Cache State:")
    # print("Cache:", cache.cache)
    print(f"Hit Rate: {cache.hit_rate()}")
