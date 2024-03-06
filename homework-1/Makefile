#Define compile
CC = g++

#Directories (For this homework does not requires, include, src, or obj)

#Compiler flags
CFLAGS = -O3
CFLAGS += -std=c++11

#The build target
all: matmul_recursive

matmul_recursive: main.cpp
	$(CC) $< $(CFLAGS) -o $@

clean:
	rm -rf matmul_recursive