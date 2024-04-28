#!/bin/bash

# Delete files if already exist
rm -rf "docs/stencil_v2_timings.csv"
rm -rf "docs/stencil_v2_timings.txt"

# Files to be written to
output_csv="docs/stencil_v1_timings.csv"
output_txt="docs/stencil_v1_timings.txt"

# Write headers to CSV file
echo "Block Size, Stencil Kernel Avg Time (ms), FLOP Count" > "$output_csv"

for i in {1..10}; do
    # Calculate for blocks of size 2^i
    blockSize=$((2**i))

    # Run nvprof with stencil_v1 executable and the current block size to get time of kernel
    time_output=$(nvprof ./stencil_v2 $blockSize 2>&1)

    # Run nvprof with stencil_v1 executable and the current block size to get flops
    flop_output=$(nvprof --metrics flop_count_sp ./stencil_v2 $blockSize 2>&1)

    # Extract stencil kernel average time
    kernel_time=$(echo "$time_output" | grep -P "stencil\(int, float\*, float const \*\)" | awk '{print $(NF-6)}')

    # Extract unit and convert if necessary
    if echo "$kernel_time" | grep -q 'us'; then
        # Convert microseconds to milliseconds
        kernel_time_ms=$(echo "$kernel_time" | sed 's/us//' | awk '{printf "%.3f", $1 / 1000}')
    elif echo "$kernel_time" | grep -q 'ms'; then
        # Just remove the 'ms' suffix
        kernel_time_ms=$(echo "$kernel_time" | sed 's/ms//')
    else
        # Assume the value is already in milliseconds
        kernel_time_ms=$kernel_time
    fi

    kernel_time=$kernel_time_ms

    # Extract single precision flop count
    flop_count=$(echo "$flop_output" | grep -A 1 "flop_count_sp" | grep -v "Metric" | awk '{print $8}')

    # Append values to CSV
    echo "$blockSize, $kernel_time, $flop_count" >> "$output_csv"
    
    # Append values to the .txt file
    echo "Block Size: $blockSize, Stencil Kernel Avg Time: $kernel_time ms, FLOP Count: $flop_count" >> "$output_txt"
done
