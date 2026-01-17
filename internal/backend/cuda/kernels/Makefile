# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -arch=sm_86 --compiler-options '-fPIC' -O3
# Use sm_86 for RTX 3070, change based on your GPU:
# sm_75 = Turing (RTX 20xx)
# sm_86 = Ampere (RTX 30xx)
# sm_89 = Ada Lovelace (RTX 40xx)

CUDA_PATH = /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lcublas -lcudart

# Output
LIB_NAME = libcuda.a
SHARED_LIB = libcuda.so

# Source files - auto-discovered recursively
SRCS := $(shell find . -type f -name '*.cu')
OBJECTS := $(SRCS:.cu=.o)

# Build static library
all: $(LIB_NAME)

$(LIB_NAME): $(OBJECTS)
	@ar rcs $(LIB_NAME) $(OBJECTS)
	@echo "Static library built: $(LIB_NAME)"

# Build shared library (optional)
shared: $(OBJECTS)
	$(NVCC) -shared -o $(SHARED_LIB) $(OBJECTS) $(LIBS)
	@echo "Shared library built: $(SHARED_LIB)"

# Compile CUDA source (create output directories if needed)
%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(LIB_NAME) $(SHARED_LIB)

.PHONY: all shared clean