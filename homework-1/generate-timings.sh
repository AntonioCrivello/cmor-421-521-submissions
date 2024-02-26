#!/bin/bash

#Clean and compile
make clean
make

#If timing file already exists, delete it
rm -rf "docs/timing.txt"

#Create the time for matrix sizes of 2^i for i = 4, 5, 6, ..., 10
for ((i = 4; i <= 10; i++))
do
    matrix_size=$((2 ** i))
    echo "Timings for Matrix of size m x n = $matrix_size" >> "docs/timing.txt"

    ./matmul_recursive $matrix_size >> "docs/timing.txt"

    echo -e "\n" >> "docs/timing.txt"
done