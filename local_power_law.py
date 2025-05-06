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
    return [data[i] for i in sampled_indices]

def power_law_with_hotspot(data, total_length=1500, exponent=1.0, 
                           window_size=50, hotspot_ratio=0.1, hotspot_boost=10):
    num_windows = total_length // window_size
    result = []
    num_elements = len(data)
    values = np.arange(1, num_elements + 1)
    base_prob = values ** -exponent
    base_prob /= base_prob.sum()

    for _ in range(num_windows):
        hotspot_indices = np.random.choice(num_elements, size=max(1, int(hotspot_ratio * window_size)), replace=False)
        prob = base_prob.copy()
        prob[hotspot_indices] *= hotspot_boost
        prob /= prob.sum()
        
        sampled_indices = np.random.choice(num_elements, size=window_size, p=prob)
        # print('hotspot_indices', hotspot_indices)
        # print('sampled_indices', sampled_indices)
        result.extend([data[i] for i in sampled_indices])

    return result

def pure_hotspot_sampling(data, sequence_length=1500, 
                          hotspot_fraction=0.1, hotspot_access_ratio=0.8):
    """
    - hotspot_fraction: 热点文档占全部文档的比例
    - hotspot_access_ratio: 最终访问序列中，热点文档占多少比例
    """
    num_elements = len(data)
    num_hotspots = max(1, int(hotspot_fraction * num_elements))

    # 随机选择固定的热点文档
    hotspot_indices = np.random.choice(num_elements, size=num_hotspots, replace=False)
    all_indices = np.arange(num_elements)
    cold_indices = np.setdiff1d(all_indices, hotspot_indices)

    num_hotspot_accesses = int(sequence_length * hotspot_access_ratio)
    num_cold_accesses = sequence_length - num_hotspot_accesses

    # 随机访问
    hotspot_samples = np.random.choice(hotspot_indices, size=num_hotspot_accesses, replace=True)
    cold_samples = np.random.choice(cold_indices, size=num_cold_accesses, replace=True)

    # 混合打乱顺序
    all_samples = np.concatenate([hotspot_samples, cold_samples])
    np.random.shuffle(all_samples)

    return [data[i] for i in all_samples], all_samples.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LRUCache and TwoQCache")
    parser.add_argument("--alpha", type=float, default=1.0, help="Exponent for power law sampling")
    parser.add_argument("--cache_size_fraction", type=float, default=0.1, help="Fraction of cache occupied by one data entry")
    parser.add_argument("--sequence_length", type=float, default=750, help="Numbers of prompts")
    args = parser.parse_args()

    alpha = float(args.alpha)
    cache_size_fraction = float(args.cache_size_fraction)
    sequence_length = int(args.sequence_length)
    
    np.random.seed(42)
    data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/142_docs.txt"
    max_size = int(668 / cache_size_fraction)
    k_value = int(max_size * 0.25)
    
    data = read_block_data_v3(data_path)[:]
    # selected_inputs = power_law_sampling(len(data),sequence_length=sequence_length, exponent=alpha)
    selected_inputs = power_law_with_hotspot(
        data, total_length=sequence_length, exponent=alpha,
        window_size=50, hotspot_ratio=0.1, hotspot_boost=10
    )
    # selected_inputs, selected_indices = pure_hotspot_sampling(
    #     data=data, 
    #     sequence_length=sequence_length, 
    #     hotspot_fraction=0.1,          # 5 个热点文档
    #     hotspot_access_ratio=0.8       # 热点访问占 80%
    # )

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
        for key, value in row:
            arc_cache.get(key)
        for key, value in reversed(row):
            arc_cache.put(key, value)
    print(f"ARCCache Hit Rate: {arc_cache.hit_rate():.2%}")

    # result_filename = f"./result/full_results_alpha_{alpha}.txt"
    # with open(result_filename, "a") as f:
    #     f.write(f"{cache_size_fraction},{lru_cache.hit_rate():.4f},{two_q_cache.hit_rate():.4f},{arc_cache.hit_rate():.4f}\n")