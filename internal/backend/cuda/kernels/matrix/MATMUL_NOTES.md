# cuBLAS Row-Major Matrix Multiplication Guide

## The Problem
cuBLAS expects column-major matrices, but we store data in row-major format.

## The Solution
For row-major C = A @ B, use the identity: C^T = B^T @ A^T

## Standard Formula

### Forward: C = A @ B
- A is [m×k] row-major
- B is [k×n] row-major  
- C is [m×n] row-major

Call: `cublasSgemm(handle, OP_N, OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n)`

Leading dimensions:
- lda (for B) = n (number of columns in row-major B[k×n])
- ldb (for A) = k (number of columns in row-major A[m×k])
- ldc = n (number of columns in row-major C[m×n])

### TransposeA: C = A^T @ B
- A is [m×k] row-major, we want A^T [k×m]
- B is [k×n] row-major
- C is [k×n] row-major

Use: C^T = B^T @ A
Call: `cublasSgemm(handle, OP_T, OP_N, n, k, m, &alpha, B, n, A, k, &beta, C, n)`

### TransposeB: C = A @ B^T
- A is [m×n] row-major
- B is [k×n] row-major, we want B^T [n×k]
- C is [m×k] row-major

Use: C^T = B @ A^T
Call: `cublasSgemm(handle, OP_N, OP_T, k, m, n, &alpha, B, n, A, n, &beta, C, k)`
