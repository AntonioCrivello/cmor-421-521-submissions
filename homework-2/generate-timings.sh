#!/bin/bash

#Clean and compile
make clean
make

# Function to save the checkpoint
save_checkpoint() {
    echo "$1 $2" > checkpoint.txt
}

# Flag indicating if the script has resumed from a checkpoint
resumed_from_checkpoint=false

# Function to resume from the last checkpoint
resume_from_checkpoint() {
    if [ -f "checkpoint.txt" ]; then
        read -r i k < checkpoint.txt
        echo "Resuming from checkpoint: i=$i, k=$k"
        resumed_from_checkpoint=true
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

    # Check for duplication only if resumed from checkpoint
    if [ "$resumed_from_checkpoint" = true ]; then
        local last_line=$(tail -n 1 "docs/timing.csv" 2>/dev/null)
        local last_matrix_size=$(echo $last_line | cut -d ',' -f1)
        local last_thread_number=$(echo $last_line | cut -d ',' -f2)
        
        # If the last entry matches the current, do not duplicate
        if [[ "$last_matrix_size" -eq "$matrix_size" && "$last_thread_number" -eq "$thread_numbers" ]]; then
            return
        fi
        
        # Reset flag after the first check
        resumed_from_checkpoint=false
    fi

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
        # Loop for thread numbers of 2^k for k = 0,...,5
        for ((; k <= 5; k++)); do
            perform_work
            save_checkpoint $i $k
        done
        k=0 # Reset k to 0 for the next loop of i
    done
    # Remove the checkpoint file after completion
    rm -f checkpoint.txt
}

# Execute the main function
main
