import numpy as np
import sys
import argparse
from tqdm import tqdm
from cache.LRU_v2 import LRUCache
from cache.two_q import TwoQCache
from cache.ARC import ARCCache
from cache.ARC_PQ import ARCCachePQ
from cache.DBL_PQ import DBLCachePQ
from cache_sequence.ARC_timestamp import ARCTimestampCache

def read_block_data_v3(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    data = [[(int(num), str(i + 1)) for num in line.split()]  # 每行作为一个子列表
            for i, line in enumerate(lines) if line]  # 过滤空行
    # 统计长度
    lengths = [len(block) for block in data]
    avg_len = sum(lengths) / len(lengths)
    print(f"📊 Total blocks: {len(data)}")
    print(f"📏 Average block length: {avg_len:.2f}")
    print(f"🔢 Min: {min(lengths)}, Max: {max(lengths)}")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LRUCache and TwoQCache")
    parser.add_argument("--cp_ratio", type=float, default=1.0, help="Exponent for power law sampling")
    cp_ratio = parser.parse_args().cp_ratio
    np.random.seed(42)
    data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/vLLM_valid.txt"
    data = read_block_data_v3(data_path)[:]
    line_lengths = [len(line) for line in data]
    print(line_lengths)
    print("Average length:", np.mean(line_lengths))
    # data = [(1, 'A'), (2, 'B'), (3, 'C'), (1, 'A1'), (4, 'D'), (5, 'E'), (1, 'A2'), (3, 'C1')]
    # selected_inputs = power_law_sampling(len(data))
    # data = selected_inputs

    # max_size = 1033
    max_size = 193.80 / 16.0 * cp_ratio     # mistral
    max_size = 11350.43 / 16.0 * cp_ratio     # SmolLM2-360M-Instruct
    max_size = 11170.23 / 16.0 * cp_ratio     # Qwen2.5-1.5B-Instruct
    print("max_size for cache:", max_size)
    k_value = int(max_size * 0.25)
    lru_cache = LRUCache(max_size=max_size)
    for idx, row in enumerate(data):
        for key, value in row:
            lru_cache.get(key)
        for key, value in reversed(row):
            lru_cache.put(key, value)
        # print(f"LRUCache Hit Rate: {idx}: {lru_cache.hit_rate():.2%}")
    print(f"LRUCache Hit Rate: {lru_cache.hit_rate():.2%}")
    
    # dbl_cache_pq = DBLCachePQ(max_size=max_size)
    # for idx, row in enumerate(tqdm(data)):
    #     for key, value in row:
    #         dbl_cache_pq.get(key)
    #     for key, value in reversed(row):
    #         dbl_cache_pq.put(key, value)
    #     # print(f"TwoQCache Hit Rate: {idx + 1}: {dbl_cache_pq.hit_rate():.2%}")
    #     # print(f"Step {idx} DBLCache A1: {len(dbl_cache_pq.A1in_data)}, Am: {len(dbl_cache_pq.Am_data)}, Hit: {dbl_cache_pq.hit_rate():.2%}")
    # print(f"DBLCache Hit Rate: {dbl_cache_pq.hit_rate():.2%}")

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
    
    # arc_pq_cache = ARCCachePQ(max_size=max_size)
    # for idx, row in enumerate(tqdm(data)):
    #     # if row[0][0] in arc_cache.T1 or row[0][0] in arc_cache.T2:
    #     #     print('hit')
    #     # else:
    #     #     print('miss')
    #     for key, value in row:
    #         arc_pq_cache.get(key)
    #     for key, value in row:
    #         arc_pq_cache.put(key, value)
    #     # print(f"Step {idx+1} ARCCache T1: {len(arc_pq_cache.T1)}, T2: {len(arc_pq_cache.T2)}, B1: {len(arc_pq_cache.B1)}, B2: {len(arc_pq_cache.B2)}, p: {arc_pq_cache.p}")
    # print(f"ARCCache Hit Rate: {arc_pq_cache.hit_rate():.2%}")
    
    arc_timestamp_cache = ARCTimestampCache(max_size=max_size)
    for seq_id, row in enumerate(tqdm(data)):
        # if row[0][0] in arc_cache.T1 or row[0][0] in arc_cache.T2:
        #     print('hit')
        # else:
        #     print('miss')
        for word_id, (key, value) in enumerate(row):
            timestamp = (seq_id, -word_id)  # 外部传入的时间戳
            arc_timestamp_cache.get(key)
        for word_id, (key, value) in enumerate(row):
            timestamp = (seq_id, -word_id)  # 外部传入的时间戳
            arc_timestamp_cache.put(key, value, timestamp)
            
        print(f"Step {seq_id+1} ARCCache T1: {len(arc_timestamp_cache.T1_data)}, T2: {len(arc_timestamp_cache.T2_data)}, B1: {len(arc_timestamp_cache.B1)}, B2: {len(arc_timestamp_cache.B2)}, p: {arc_timestamp_cache.p}")
        print(f"ARCTimestampCache Hit Rate: {arc_timestamp_cache.hit_rate():.2%}")
    
    print(f"ARCTimestampCache Hit Rate: {arc_timestamp_cache.hit_rate():.2%}")
    