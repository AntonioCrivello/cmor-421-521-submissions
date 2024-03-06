#define compiler
#CC = g++ -std=c++11
CC = g++

#directories
INC_DIR = ./include
SRC_DIR = ./src
OBJ_DIR = ./obj

$(shell mkdir -p $(OBJ_DIR))

#create list of all header, source, and object files based on the directory
INC_FILES := $(wildcard $(INC_DIR)/*.hpp)
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))

#flags
CFLAGS = -I$(INC_DIR)
CFLAGS += -O3
CFLAGS += -std=c++11

#recompile if files in ./include change
DEPS = $(INC_FILES)

#Define targets
all: matmul_recursive

matmul_recursive: $(OBJ_FILES) main.cpp
	$(CC) $^ $(CFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS)
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f matmul_recursive $(OBJ_DIR)/*.o *~ *.o
