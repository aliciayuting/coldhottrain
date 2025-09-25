#!/bin/bash

dim=16
hot_ratio=0.2
dim_thres=8192
hot_ratio_thres=1

while (( $(echo "$hot_ratio <= $hot_ratio_thres" | bc -l) )); do
    while [ $dim -le $dim_thres ]; do
        echo "================ Running benchmark with k=$dim, hot_ratio=$hot_ratio ================"
        cmd="python benchmark.py --k $dim --hot-ratio $hot_ratio --skip"
        eval "$cmd"

        cmd="python benchmark.py --k $dim --hot-ratio $hot_ratio"
        eval "$cmd"
        dim=$((dim * 2))
    done
    hot_ratio=$(echo "$hot_ratio + 0.1" | bc -l)
    dim=16
done