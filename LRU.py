from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hit_count = 0
        self.access_count = 0
    
    def get(self, key, value):
        self.access_count += 1

        if key in self.cache:
            self.hit_count += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self._put(key, value)
        return value

    def _put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)

        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def hit_rate(self):
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0


def read_block_data_v1():
    with open("../data/block_log.txt", "r") as f:
        lines = [line.strip() for line in f.readlines()]

    data = [(int(content), str(i + 1)) for i, content in enumerate(lines)]
    return data

def read_block_data_v2(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    data = [(int(num), str(i + 1)) 
            for i, line in enumerate(lines) if line
            for num in line.split()]
    return data

data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/block_log_200.txt"
data = read_block_data_v2(data_path)
# data = [(1, 'A'), (2, 'B'), (3, 'C'), (1, 'A1'), (4, 'D'), (5, 'E'), (1, 'A2'), (3, 'C1')]


cache = LRUCache(max_size=5000)     # 2383
for key, value in data:
    print(f"Accessed ({key}): {cache.get(key, value)}")

# print("\nFinal Cache State:")
print("Cache:", cache.cache)
print(f"Hit Rate: {cache.hit_rate()}")
