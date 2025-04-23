from sortedcontainers import SortedDict

# 你希望排序的依据
sorting_keys = {
    1: (3, 10),
    2: (2, 15),
    3: (3, 5)
}

# 构造 SortedDict
sorted_dict = SortedDict(lambda x: (x[0], -x[1]))  # 按第一项升序，第二项降序

# 插入数据
for k, v in sorting_keys.items():
    sorted_dict[v] = k  # 这里 key 是 (3, 10) 这样的元组，值是 1, 2, 3

print(sorted_dict)
