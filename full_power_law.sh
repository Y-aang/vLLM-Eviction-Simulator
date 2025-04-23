#!/bin/bash

# 删除旧的结果文件
ALPHA=6.0
RESULT_FILE="./result/full_results_alpha_${ALPHA}.txt"
rm -f $RESULT_FILE

# 设定不同的 cache_size_fraction
for frac in 0.01 0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.17 0.20; do
# for frac in 0.51 0.6 0.7 0.8 0.9 0.95 0.99 1.5 2.0 10.0; do
    echo "Running with cache_size_fraction=$frac"
    
    python full_power_law.py --alpha $ALPHA --cache_size_fraction $frac
done

echo "All tests completed. Results saved in $RESULT_FILE."
