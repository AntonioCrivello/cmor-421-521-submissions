#!/bin/bash

# Delete files if they already exist
rm -rf "docs/reduction_timings.csv"
rm -rf "docs/reduction_timings.txt"

# Files to be written to
output_csv="docs/reduction_timings.csv"
output_txt="docs/reduction_timings.txt"

# Write headers to CSV files
echo "Version, Kernel Avg Time (ms), FLOP Count" > "$output_csv"


# Process each version
for version in v0 v1 v2; do
    # Compile reduction files
    nvcc reduction_$version.cu -o reduction_$version

    # Run nvprof to get time of kernel
    time_output=$(nvprof ./reduction_$version 2>&1)

    # Run nvprof to get flops
    flop_output=$(nvprof --metrics flop_count_sp ./reduction_$version 2>&1)

    # Extract kernel average time
    kernel_time=$(echo "$time_output" | grep -P "partial_reduction\(int, float\*, float const \*\)" | awk '{print $(NF-6)}')

    # Convert units if necessary
    if echo "$kernel_time" | grep -q 'us'; then
        kernel_time_ms=$(echo "$kernel_time" | sed 's/us//' | awk '{printf "%.3f", $1 / 1000}')
    elif echo "$kernel_time" | grep -q 'ms'; then
        kernel_time_ms=$(echo "$kernel_time" | sed 's/ms//')
    else
        kernel_time_ms=$kernel_time
    fi

    # Extract single precision flop count
    flop_count=$(echo "$flop_output" | grep -A 1 "flop_count_sp" | grep -v "Metric" | awk '{print $8}')

    # Append values to CSV
    echo "$version, $kernel_time_ms, $flop_count" >> "$output_csv"
    
    # Append values to the .txt file
    echo "$version: Kernel Avg Time: $kernel_time_ms ms, FLOP Count: $flop_count" >> "$output_txt"
done
