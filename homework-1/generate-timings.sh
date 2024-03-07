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
        read -r i k <<< "$(cat checkpoint.txt)"
        echo "Resuming from checkpoint: i=$i, k=$k"
    else
        i=4
        k=2
        echo "No checkpoint found. Starting from the beginning: i=$i, k=$k"
        rm -rf "docs/timing.csv"
        rm -rf "docs/results.txt"
    fi
}

# Function to perform the actual work
perform_work() {
    matrix_size=$((2 ** i))
    block_size=$((2 ** k))
    output=$(./matmul_recursive $matrix_size $block_size)

    # Extract the timings for each matrix-matrix multiplication type
    naive_timing=$(echo "$output" | tail -n 4 | head -n 1 | awk '{print $NF}')
    blocked_timing=$(echo "$output" | tail -n 3 | head -n 1 | awk '{print $NF}')
    recursive_timing=$(echo "$output" | tail -n 2 | head -n 1 | awk '{print $NF}')
    recursive_intermediate_timing=$(echo "$output" | tail -n 1 | awk '{print $NF}')

    # Output the timings and sizing to .csv for plotting
    echo "$matrix_size, $block_size, $naive_timing, $blocked_timing, $recursive_timing, $recursive_intermediate_timing" >> "docs/timing.csv"
    # Create a .txt file with the results of the program
    echo "$output" >> "docs/results.txt"
}

# Main function
main() {
    resume_from_checkpoint
    # Create the time for matrix sizes of 2^i for i = 4, 5, 6, ..., 10
    for ((; i <= 10; i++)); do
        for ((; k <= i; k++)); do
            if [ "$k" -gt 8 ]; then
                break
            fi
            perform_work
            save_checkpoint $i $k
        done
        k=2
    done
    # Remove the checkpoint file after completion
    rm -f checkpoint.txt
}

# Execute the main function
main


# #!/bin/bash

# #Clean and compile
# make clean
# make

# # If timing files already exists, delete it
# rm -rf "docs/timing.csv"
# rm -rf "docs/results.txt"

# # CSV header
# echo "Matrix Size, Block Size, Naive Timing, Blocked Timing, Recursive Timing, Recursive Intermediate Timing" >> "docs/timing.csv"

# # Create the time for matrix sizes of 2^i for i = 4, 5, 6, ..., 10

# for ((i = 4; i <= 10; i++)); do
#     for ((k = 2; k <= i; k++)); do
#         if [ "$k" -gt 8 ]; then
#             break
#         fi
#         matrix_size=$((2 ** i))
#         block_size=$((2 ** k))
#         output=$(./matmul_recursive $matrix_size $block_size)

#         #Extract the timings for each matrix-matrix multiplication type
#         naive_timing=$(echo "$output" | tail -n 4 | head -n 1 | awk '{print $NF}') 
#         blocked_timing=$(echo "$output" | tail -n 3 | head -n 1 | awk '{print $NF}')
#         recursive_timing=$(echo "$output" | tail -n 2 | head -n 1 | awk '{print $NF}')
#         recursive_intermediate_timing=$(echo "$output" | tail -n 1 | awk '{print $NF}')

#         #Output the timings and sizing to .csv for plotting
#         echo "$matrix_size, $block_size, $naive_timing, $blocked_timing, $recursive_timing, $recursive_intermediate_timing" >> "docs/timing.csv"
#         #Creating a .txt file with the results of the program
#         echo "$output" >> "docs/results.txt"
#     done
# done
