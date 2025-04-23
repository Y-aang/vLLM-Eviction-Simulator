import numpy as np
import argparse
from LRU_v2 import LRUCache
from two_q import TwoQCache

def read_block_data_lines(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 每行代表一个 block，按空格切分成 int，然后赋一个 dummy value（如字符串行号）
    data = [[(int(num), str(i + 1)) for num in line.split()] for i, line in enumerate(lines)]
    # 统计长度
    lengths = [len(block) for block in data]
    avg_len = sum(lengths) / len(lengths)
    print(f"📊 Total blocks: {len(data)}")
    print(f"📏 Average block length: {avg_len:.2f}")
    print(f"🔢 Min: {min(lengths)}, Max: {max(lengths)}")
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LRUCache and TwoQCache")
    parser.add_argument("--alpha", type=float, default=1.0, help="(ignored) kept for compatibility")
    parser.add_argument("--cache_size_fraction", type=float, default=0.1, help="Fraction of cache occupied by one data entry")
    args = parser.parse_args()

    cache_size_fraction = args.cache_size_fraction
    np.random.seed(42)

    data_path = "/Users/shenyang/Desktop/MS Research/workplace/data/meg_docqa.txt"
    
    # 直接读取每一行作为一个 block
    data = read_block_data_lines(data_path)

    max_size = int(20.5 / cache_size_fraction)
    k_value = int(max_size * 0.25)

    # 直接顺序处理 data
    lru_cache = LRUCache(max_size=max_size)
    for row in data:
        for key, value in row:
            lru_cache.get(key)
        for key, value in reversed(row):
            lru_cache.put(key, value)
        # print(f"LRUCache Hit Rate: {lru_cache.hit_rate():.2%} {lru_cache.hit_count} {lru_cache.access_count}")

    print(f"LRUCache Hit Rate: {lru_cache.hit_rate():.2%}")

    two_q_cache = TwoQCache(max_size=max_size, k=k_value)
    for row in data:
        for key, value in row:
            two_q_cache.get(key)
        for key, value in reversed(row):
            two_q_cache.put(key, value)

    print(f"TwoQCache Hit Rate: {two_q_cache.hit_rate():.2%}")

    result_filename = f"./result/results_alpha_{args.alpha}.txt"
    with open(result_filename, "a") as f:
        f.write(f"{cache_size_fraction},{lru_cache.hit_rate():.4f},{two_q_cache.hit_rate():.4f}\n")
