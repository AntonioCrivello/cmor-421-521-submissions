#!/bin/bash

#Clean and compile
make clean
make

# Function to save the checkpoint
save_checkpoint() {
    echo "$1 $2" > checkpoint.txt
}

# Function to resume from the last checkpoint
resume_from_checkpoint() {
    if [ -f "checkpoint.txt" ]; then
        read -r i k < checkpoint.txt
        echo "Resuming from checkpoint: i=$i, k=$k"
    else
        i=4
        k=0
        echo "No checkpoint found. Starting from the beginning: i=$i, k=$k"
        > "docs/timing.csv"
        > "docs/results.txt"
    fi
}

# Function to perform the actual work
perform_work() {
    local matrix_size=$((2 ** i))
    local thread_numbers=$((2 ** k))
    local output=$(./openmp_mat $matrix_size $thread_numbers)

    # Extract the timings for each matrix-matrix multiplication type
    local matmul_parallel=$(echo "$output" | tail -n 2 | head -n 1 | awk '{print $NF}')
    local matmul_parallel_collapsed=$(echo "$output" | tail -n 1 | awk '{print $NF}')
    # recursive_timing=$(echo "$output" | tail -n 2 | head -n 1 | awk '{print $NF}')
    # recursive_intermediate_timing=$(echo "$output" | tail -n 1 | awk '{print $NF}')

    # Output the timings and sizing to .csv for plotting
    echo "$matrix_size, $thread_numbers, $matmul_parallel, $matmul_parallel_collapsed" >> "docs/timing.csv"

    # Append the results to .txt for documentation
    echo -e "Matrix Size: $matrix_size, Thread Number: $thread_numbers\n$output" >> "docs/results.txt"
}

# Main function
main() {
    resume_from_checkpoint
    # Loop for matrix sizes of 2^i for i = 4, 5, 6,...,10
    for ((; i <= 10; i++)); do
        # Loop for thread numbers of 2^k for k = 1,...,5
        for ((; k <= 5; k++)); do
            perform_work
            save_checkpoint $i $k

        done
        k=0 # Reset k to 1 for the next loop of i
    done
    # Remove the checkpoint file after completion
    rm -f checkpoint.txt
}

# Execute the main function
main


