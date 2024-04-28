#!/bin/bash

# Delete files if already exist
rm -rf "docs/timing.csv"
rm -rf "docs/results.txt"

# Files to be written to
output_csv="docs/stencil_v1_timings.csv"
output_txt="docs/stencil_v1_timings.txt"

# Write headers to CSV file
echo "Block Size, Stencil Kernel Avg Time (ms), FLOP Count" > "$output_csv"

for i in {1..8}; do
    # Calculate for blocks of size 2^i
    blockSize=$((2**i))

    # Run nvprof with stencil_v1 executable and the current block size
    output=$(nvprof --print-gpu-summar --metrics flop_count_sp ./stencil_v1 $blockSize)

    # Extract single precision flop count
    flop_count=$(echo "$output" | grep -P "flop_count_sp\s+" | awk '{print $(NF-1)}')

    # Extract stencil kernel average time
    kernel_time=$(echo "$output" | grep -P "stencil\(int, float\*, float const \*\)" | awk '{print $4}')

    # Append values to CSV
    echo "$blockSize, $kernel_time, $flop_count" >> "$output_csv"
    
    # Append values to the .txt file
    echo "Block Size: $blockSize, Stencil Kernel Avg Time: $kernel_time, FLOP Count: $flop_count >> "$output.txt"
}