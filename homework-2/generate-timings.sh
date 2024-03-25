#!/bin/bash

# Clean and compile
make clean
make

# Set block size
block_size=8

# If timing files already exists, delete it
rm -rf "docs/timing.csv"
rm -rf "docs/results.txt"

# Output files
output_csv="timings.csv"
output_txt="timings.txt"

# Clear the output files to start fresh
echo "Matrix Size,Block Size,Number of Threads,Elapsed Back Solve,Elapsed Parallel Back Solve Static,Elapsed Parallel Back Solve Dynamic" > $output_csv
echo "Timings for various matrix sizes and number of threads" > $output_txt

# Loop over matrix sizes
for matrix_size in 16 32 64 128 256 512 1024 2048 4096 8192 16384; do
    # Loop over number of threads
    for num_threads in 1 2 4 8 16 32 64; do
        # Run the executable and capture the output
        output=$(./openmp_mat $matrix_size $block_size $num_threads)
    

        elapsedBackSolve=$(echo "$output" | tail -n 3 | head -n 1 | awk '{print $NF}')
        elapsedParallelBackSolveStatic=$(echo "$output" | tail -n 2 | head -n 1 | awk '{print $NF}')
        elapsedParallelBackSolveDynamic=$(echo "$output" | tail -n 1 | awk '{print $NF}')

        
        # Write to CSV
        echo "$matrix_size,$block_size,$num_threads,$elapsedBackSolve,$elapsedParallelBackSolveStatic,$elapsedParallelBackSolveDynamic" >> $output_csv
        
        # Write to TXT
        echo "Matrix Size: $matrix_size, Block Size: $block_size, Number of Threads: $num_threads" >> $output_txt
        echo "Elapsed Back Solve: $elapsedBackSolve" >> $output_txt
        echo "Elapsed Parallel Back Solve Static: $elapsedParallelBackSolveStatic" >> $output_txt
        echo "Elapsed Parallel Back Solve Dynamic: $elapsedParallelBackSolveDynamic" >> $output_txt
        echo "" >> $output_txt
    done
done
