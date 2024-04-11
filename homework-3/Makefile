# Define compiler
CC = mpic++

# Directories
INC_DIR = ./include
SRC_DIR = ./src 

# Create automatic list of all header and source files based on directory
INC_FILES := $(wildcard $(INC_DIR)/*.hpp)
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)

# Flags
CFLAGS = -I$(INC_DIR)

# Dependicies 
DEPS = $(INC_FILES)

# Define targets
all: summa

summa: $(OBJ_FILES) summa.cpp 
		