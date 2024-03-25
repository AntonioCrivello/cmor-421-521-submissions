#Define compile
# CC = g++-13
CC = g++

#Directories (For this homework does not requires, include, src, or obj)

#Compiler flags
# CFLAGS = -O3
CFLAGS = -fopenmp

#The build target
all: openmp_mat

openmp_mat: main.cpp
	$(CC) $^ $(CFLAGS) -o $@

clean:
	rm -rf openmp_mat