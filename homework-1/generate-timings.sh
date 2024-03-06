#!/bin/bash

#Clean and compile
make clean
make

# If timing file already exists, delete it
rm -rf "docs/timing.csv"

# CSV header
echo "Matrix Size, Block Size, Naive Timing, Blocked Timing, Recursive Timing" >> "docs/timing.csv"

# Create the time for matrix sizes of 2^i for i = 4, 5, 6, ..., 10

for ((i = 4; i <= 10; i++)); do
    for ((k = 2; k <= i; k++)); do
        matrix_size=$((2 ** i))
        block_size=$((2 ** k))
        output=$(./matmul_recursive $matrix_size $block_size)

        #Extract the timings for each matrix-matrix multiplication type
        naive_timing=$(echo "$output" | tail -n 3 | head -n 1 | awk '{print $NF}') 
        blocked_timing=$(echo "$output" | tail -n 2 | head -n 1 | awk '{print $NF}')
        recursive_timing=$(echo "$output" | tail -n 1 | awk '{print $NF}')

        #Output the timings and sizing to .csv for plotting
        echo "$matrix_size, $block_size, $naive_timing, $blocked_timing, $recursive_timing" >> "docs/timing.csv"
        #Creating a .txt file with the results of the program
        echo "$output" >> "docs/results.txt"
    done
done
