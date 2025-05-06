import numpy as np
import sys
import argparse
from tqdm import tqdm
from cache.LRU_v2 import LRUCache
from cache.two_q import TwoQCache
from cache.ARC import ARCCache
from cache.DBL import DBLCache
from cache.DBL_PQ import DBLCachePQ
from cache.LFU import LFUCache

def read_block_data_v3(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    data = [[(int(num), str(i + 1)) for num in line.split()]  # 每行作为一个子列表
            for i, line in enumerate(lines) if line]  # 过滤空行
    # 统计长度
    # lengths = [len(block) for block in data]
    # avg_len = sum(lengths) / len(lengths)
    # print(f"📊 Total blocks: {len(data)}")
    # print(f"📏 Average block length: {avg_len:.2f}")
    # print(f"🔢 Min: {min(lengths)}, Max: {max(lengths)}")
    
    return data

def power_law_sampling(num_elements, sequence_length=1500, exponent=1.0):
    values = np.arange(1, num_elements + 1)
    probabilities = values ** -exponent
    probabilities /= probabilities.sum()
    sampled_indices = np.random.choice(values - 1, size=sequence_length, p=probabilities)
    print('sampled_indices', sampled_indices)
    return [data[i] for i in sampled_indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LRUCache and TwoQCache")
    parser.add_argument("--alpha", type=float, default=1.0, help="Exponent for power law sampling")
    parser.add_argument("--cache_size_fraction", type=float, default=0.1, help="Fraction of cache occupied by one data entry")
    parser.add_argument("--sequence_length", type=float, default=150, help="Numbers of prompts")    # 750
    args = parser.parse_args()

    alpha = float(args.alpha)
    cache_size_fraction = float(args.cache_size_fraction)
    sequence_length = int(args.sequence_length)
    
    np.random.seed(42)
    data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/142_docs.txt"
    max_size = int(668 / cache_size_fraction)
    k_value = int(max_size * 0.25)
    
    data = read_block_data_v3(data_path)[:]
    selected_inputs = power_law_sampling(len(data),sequence_length=sequence_length, exponent=alpha)
    data = selected_inputs

    lru_cache = LRUCache(max_size=max_size)
    for row in data:
        for key, value in row:
            lru_cache.get(key)
        for key, value in reversed(row):
            lru_cache.put(key, value)
    print(f"LRUCache Hit Rate: {lru_cache.hit_rate():.2%}")
    
    dbl_cache = DBLCachePQ(max_size=max_size)
    for idx, row in enumerate(data):
        for key, value in row:
            dbl_cache.get(key)
        for key, value in reversed(row):
            dbl_cache.put(key, value)
        # print(f"Step {idx} DBLCache A1: {len(dbl_cache.A1in_data)}, Am: {len(dbl_cache.Am_data)}")
    print(f"DBLCache Hit Rate: {dbl_cache.hit_rate():.2%}")
    
    lfu_cache = LFUCache(max_size=max_size)
    for idx, row in enumerate(data):
        for key, value in row:
            lfu_cache.get(key)
        for key, value in reversed(row):
            lfu_cache.put(key, value)
        # print(f"Step {idx} DBLCache A1: {len(lfu_cache.A1in_data)}, Am: {len(lfu_cache.Am_data)}")
        # print(f"Step {idx} lfu_cache.min_freq {lfu_cache.min_freq}")
    print(f"LFUCache Hit Rate: {lfu_cache.hit_rate():.2%}")
    
    # dbl_cache_pq = DBLCachePQ(max_size=max_size)
    # for row in data:
    #     for key, value in row:
    #         dbl_cache_pq.get(key)
    #     for key, value in reversed(row):
    #         dbl_cache_pq.put(key, value)
    # print(f"DBLCachePQ Hit Rate: {dbl_cache_pq.hit_rate():.2%}")

    two_q_cache = TwoQCache(max_size=max_size, k=k_value)
    for row in data:
        for key, value in row:
            two_q_cache.get(key)
        for key, value in reversed(row):
            two_q_cache.put(key, value)
    print(f"TwoQCache Hit Rate: {two_q_cache.hit_rate():.2%}")
    
    arc_cache = ARCCache(max_size=max_size)
    for idx, row in enumerate(tqdm(data)):
        # if row[0][0] in arc_cache.T1 or row[0][0] in arc_cache.T2:
        #     print('hit')
        # else:
        #     print('miss')
        for key, value in row:
            arc_cache.get(key)
        for key, value in reversed(row):
            arc_cache.put(key, value)
        # print(f"Step {idx+1} ARCCache T1: {len(arc_cache.T1)}, T2: {len(arc_cache.T2)}, B1: {len(arc_cache.B1)}, B2: {len(arc_cache.B2)}, p: {arc_cache.p}")
    print(f"ARCCache Hit Rate: {arc_cache.hit_rate():.2%}")

    # result_filename = f"./result/full_results_alpha_{alpha}.txt"
    # with open(result_filename, "a") as f:
    #     f.write(f"{cache_size_fraction},{lru_cache.hit_rate():.4f},{two_q_cache.hit_rate():.4f},{arc_cache.hit_rate():.4f}\n")