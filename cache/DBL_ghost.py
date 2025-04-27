from collections import OrderedDict, deque
import numpy as np

class DBLCache:
    def __init__(self, max_size):
        # The only constraint: total length of 2LRU smaller than max_size
        self.k = int(max_size * 0.5)  # size of A1in and A1out
        self.max_size = max_size    # total size of 2Queue
        self.A1in = OrderedDict()
        self.A1out = deque(maxlen=self.k)
        self.Am = OrderedDict()     # Long-term Main Queue
        self.hit_count = 0
        self.access_count = 0
    
    def get(self, key):
        self.access_count += 1

        if key in self.Am:
            self.hit_count += 1
            # self.Am.move_to_end(key)
            return self.Am[key]

        if key in self.A1in:
            self.hit_count += 1
            # self.A1in.move_to_end(key)
            return self.A1in[key]
            # value = self.A1in.pop(key)
            # self._promote_to_Am(key, value)
            # return self.Am[key]

        return None 
    
    def put(self, key, value):
        if key in self.Am:
            self.Am[key] = value
            self.Am.move_to_end(key)
            return

        if key in self.A1in:
            self.A1in[key] = value
            # self.A1in.move_to_end(key)      # Define LRU or FIFO for A1in
            value = self.A1in.pop(key)      # define whether it's a ARC-2Q or traditional-2Q
            self._promote_to_Am(key, value)
            return

        if key in self.A1out:           # [ghost cache]
            assert len(self.A1in) + len(self.Am) <= self.max_size
            if len(self.A1in) + len(self.Am) == self.max_size:
                self.Am.popitem(last=False)
            self.A1out.remove(key)
            self._promote_to_Am(key, value)

        assert len(self.A1in) <= self.k
        if len(self.A1in) == self.k:
            old_key, _ = self.A1in.popitem(last=False)
            self.A1out.append(old_key)  # [ghost cache]
            self.A1in[key] = value
        else:
            assert len(self.A1in) + len(self.Am) <= self.max_size
            if len(self.A1in) + len(self.Am) == self.max_size:
                self.Am.popitem(last=False)
            self.A1in[key] = value
            
            
    def _promote_to_Am(self, key, value):
        # if len(self.Am) >= self.max_size - self.k:
        #     self.Am.popitem(last=False)
        self.Am[key] = value
        self.Am.move_to_end(key)

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

    cache = DBLCache(max_size=600 * 10)     # 2383
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
