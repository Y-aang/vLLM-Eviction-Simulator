from collections import defaultdict, OrderedDict
import numpy as np

class LFUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = {}                      # key -> (value, freq)
        self.freq_table = defaultdict(OrderedDict)  # freq -> OrderedDict of keys
        self.min_freq = 0
        self.hit_count = 0
        self.access_count = 0

    def get(self, key):
        self.access_count += 1
        if key in self.data:
            self.hit_count += 1
            return self.data[key][0]  # return value only
        return None

    def put(self, key, value):
        if key in self.data:
            # Update value only; increase frequency
            old_value, freq = self.data[key]
            new_freq = freq + 1
            self.data[key] = (value, new_freq)

            # Move key from old freq to new freq
            del self.freq_table[freq][key]
            if not self.freq_table[freq]:
                del self.freq_table[freq]
                if freq == self.min_freq:
                    self.min_freq += 1
            self.freq_table[new_freq][key] = None
            return

        if len(self.data) >= self.max_size:
            self._evict()

        # Insert new key
        self.data[key] = (value, 1)
        self.freq_table[1][key] = None
        self.min_freq = 1

    def _evict(self):
        # Remove the least frequently used key (lowest freq, oldest in that freq)
        freq = self.min_freq
        key, _ = self.freq_table[freq].popitem(last=False)
        if not self.freq_table[freq]:
            del self.freq_table[freq]
        del self.data[key]

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
    selected_inputs = power_law_sampling(len(data))
    data = selected_inputs

    cache = LFUCache(max_size=600 * 10)
    for i, row in enumerate(data):
        for key, value in row:
            cache.get(key)
        for key, value in reversed(row):
            cache.put(key, value)

    print(f"Hit Rate: {cache.hit_rate():.2%}")
