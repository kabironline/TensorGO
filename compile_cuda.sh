#!/bin/bash

cd internal/backend/cuda/kernels

# Compiling Kernels using MAKEFILE
make clean
make all -j8

# print success message
echo "CUDA Kernels Compiled Successfully!"