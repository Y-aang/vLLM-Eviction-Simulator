import numpy as np
import sys
import argparse
from cache.LRU_v2 import LRUCache
from cache.two_q import TwoQCache
from cache.ARC import ARCCache
from cache.DBL_PQ import DBLCachePQ

def read_block_data_v3(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    data = [[(int(num), str(i + 1)) for num in line.split()]  # 每行作为一个子列表
            for i, line in enumerate(lines) if line]  # 过滤空行

    return data

if __name__ == "__main__":
    np.random.seed(42)
    data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/vLLM_valid.txt"
    data = read_block_data_v3(data_path)[:]
    line_lengths = [len(line) for line in data]
    print(line_lengths)
    print("Average length:", np.mean(line_lengths))
    # data = [(1, 'A'), (2, 'B'), (3, 'C'), (1, 'A1'), (4, 'D'), (5, 'E'), (1, 'A2'), (3, 'C1')]
    # selected_inputs = power_law_sampling(len(data))
    # data = selected_inputs

    max_size = 1033
    k_value = int(max_size * 0.25)
    lru_cache = LRUCache(max_size=max_size)
    for idx, row in enumerate(data):
        for key, value in row:
            lru_cache.get(key)
        for key, value in reversed(row):
            lru_cache.put(key, value)
        # print(f"LRUCache Hit Rate: {idx}: {lru_cache.hit_rate():.2%}")
    print(f"LRUCache Hit Rate: {lru_cache.hit_rate():.2%}")
    
    dbl_cache_pq = DBLCachePQ(max_size=max_size)
    for idx, row in enumerate(data):
        for key, value in row:
            dbl_cache_pq.get(key)
        for key, value in reversed(row):
            dbl_cache_pq.put(key, value)
        # print(f"TwoQCache Hit Rate: {idx + 1}: {dbl_cache_pq.hit_rate():.2%}")
        # print(f"Step {idx} DBLCache A1: {len(dbl_cache_pq.A1in_data)}, Am: {len(dbl_cache_pq.Am_data)}, Hit: {dbl_cache_pq.hit_rate():.2%}")
    print(f"DBLCache Hit Rate: {dbl_cache_pq.hit_rate():.2%}")

    # two_q_cache = TwoQCache(max_size=max_size, k=k_value)
    # for idx, row in enumerate(data):
    #     for key, value in row:
    #         two_q_cache.get(key)
    #     for key, value in reversed(row):
    #         two_q_cache.put(key, value)
    #     # print(f"TwoQCache Hit Rate: {idx + 1}: {two_q_cache.hit_rate():.2%}")
    # print(f"TwoQCache Hit Rate: {two_q_cache.hit_rate():.2%}")
        
        # if flag==True:
        #     print('evict happen')
        # print(f"{i} round - Hit Rate: {cache.hit_rate()} Hit count: {cache.hit_count} Access count: {cache.access_count}")

    # # print("\nFinal Cache State:")
    # print("Cache:", cache.cache)
    # print(f"Hit Rate: {cache.hit_rate()}")
