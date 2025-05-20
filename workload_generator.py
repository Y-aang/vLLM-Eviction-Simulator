import random

def generate_padded_hash_file(output_path, num_docs=100, avg_length=10, jitter=0.2, hash_width=19):
    """
    生成模拟 content hash 文件，每个 hash 是固定宽度的字符串数字，从0开始递增。

    参数:
    - output_path: 输出文件路径
    - num_docs: 行数（文档数量）
    - avg_length: 每行平均 hash 数量
    - jitter: 控制每行长度的波动范围（0.2 表示 ±20%）
    - hash_width: 每个 hash 的最小位数（如19表示从 0000000000000000000 开始）
    """
    current_id = 0

    with open(output_path, "w") as f:
        for _ in range(num_docs):
            length = random.randint(
                max(1, int(avg_length * (1 - jitter))),
                int(avg_length * (1 + jitter))
            )
            hashes = [str(current_id + i).zfill(hash_width) for i in range(length)]
            current_id += length
            f.write(" ".join(hashes) + "\n")

# 示例调用
generate_padded_hash_file("/Users/shenyang/Desktop/MS Research/workplace/data/artificial_docs.txt", num_docs=10000, avg_length=668)
