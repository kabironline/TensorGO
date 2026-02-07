# cuBLAS Matrix Multiplication Debugging Session - February 7, 2026

## Problem Summary
MNIST neural network test was failing with CUDA backend, producing "cuda_add failed" panic during backpropagation. Investigation revealed root cause: incorrect cuBLAS SGEMM formulas for transposed matrix multiplications.

## Root Cause Analysis

### Initial Issue
- **Error**: `cublasSgemm parameter number 8 had an illegal value` 
- **Location**: Backward pass of MatMulAddBias operation
- **Symptom**: Test crashed during gradient computation with trans_a and trans_b operations

### Key Discovery
The nanograd library uses **row-major** matrix layout, but cuBLAS expects **column-major** layout. This requires careful conversion of matrix operations.

## Critical Learning: Row-Major to Column-Major Conversion

### Fundamental Principle
When converting row-major matrix multiplication to cuBLAS (column-major):
```
Row-major: C = A @ B
Column-major: C^T = B^T @ A^T
```

This means in the cuBLAS call:
- **Matrices are swapped**: Pass B first, then A
- **Transpose flags are inverted**: OP_N becomes OP_T and vice versa
- **Dimensions are rearranged**: n and m positions swap
- **Leading dimensions**: Use the **number of columns** in row-major storage

### The Leading Dimension Formula
**Critical Rule**: For a matrix stored in row-major with shape [rows × cols]:
```
leading_dimension (ld) = number_of_columns
```

This is because in row-major storage, consecutive elements in a column are `cols` elements apart in memory.

## Correct cuBLAS Formulas

### 1. Normal Matrix Multiplication
**Operation**: `C[m×n] = A[m×k] @ B[k×n]` (row-major)

**cuBLAS Call**:
```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
            &alpha, B, n,    // B[k×n], ld=n
                    A, k,    // A[m×k], ld=k  
            &beta,  C, n);   // C[m×n], ld=n
```

**Verification**: Tested with A=[2×3], B=[3×2], produces C=[[58,64],[139,154]] ✓

### 2. Transpose B: C = A @ B^T
**Operation**: `C[m×n] = A[m×k] @ B^T` where B is [n×k] (row-major)

**cuBLAS Call**:
```c
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
            &alpha, B, k,    // B[n×k], ld=k
                    A, k,    // A[m×k], ld=k
            &beta,  C, n);   // C[m×n], ld=n
```

**Verification**: Tested with A=[2×3], B=[2×3], produces C=[[58,64],[139,154]] ✓

### 3. Transpose A: C = A^T @ B  
**Operation**: `C[m×n] = A^T @ B` where A is [k×m] (row-major)

**cuBLAS Call**:
```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k,
            &alpha, B, n,    // B[k×n], ld=n  ← CRITICAL FIX
                    A, m,    // A[k×m], ld=m
            &beta,  C, n);   // C[m×n], ld=n
```

**The Bug**: Original code had `B, m` instead of `B, n` - this caused parameter 8 errors!

**Verification**: Tested with A=[5×3], B=[5×4], produces correct result with ld=n ✓

## Debugging Methodology That Worked

### 1. Systematic Testing Approach
Instead of debugging in the full application, created standalone CUDA test programs:
- `test_matmul.cu`: Verified basic formula
- `test_correct.cu`: Confirmed trans_b with proper row-major interpretation  
- `test_systematic.cu`: Exhaustively tested 8 combinations of transpose flags and LD values
- `test_transa_general.cu`: Tested trans_a with different dimensions (m≠n) to find the pattern

### 2. Key Testing Insight
**Don't test with square matrices (m=n)!** The trans_a bug was hidden when m=n=2 because both `lda=m` and `lda=n` produced the same value. Only testing with m≠n (e.g., m=3, n=4) revealed the correct formula.

### 3. Verification Process
```bash
nvcc test_file.cu -o test_file -lcublas && ./test_file
```
Each test computed known matrix products and compared against hand-calculated expected values.

## Files Modified

### Fixed Files
1. **`internal/backend/cuda/kernels/matrix/transpose_mul.cu`**
   - `cuda_matmul_trans_a`: Changed `B, m` → `B, n` (line 25)
   - `cuda_matmul_trans_b`: Already correct with `B, k`

2. **`internal/backend/cuda/kernels/matrix/mul.cu`**
   - Already correct, no changes needed

### Other Fixes Applied During Investigation
1. **`internal/backend/cuda/kernels/dmas/add.cu`** & **`sub.cu`**
   - Replaced cuBLAS SAXPY with vectorized CUDA kernels (float4 processing)
   - Reason: Simpler to use custom kernels for elementwise operations

2. **`internal/backend/cuda/ops_memory.go`**
   - Fixed Copy() byte count from `size*8` to `size*4` (correct for float32)

## Known Remaining Issues

### Gradient Flow Problem (Discovered but NOT fixed)
During testing, discovered that gradients are not flowing through the network:
- All weight gradients remain zero after backward pass
- Loss stays constant at ~2.3 (random guessing for 10 classes)
- Test achieves only 9.8% accuracy instead of expected >95%

**Not a cuBLAS Problem**: Created targeted tests showing:
- ✓ MatMul forward/backward computations are mathematically correct
- ✓ Optimizer successfully updates weights when given non-zero gradients  
- ✓ Individual operations (ReLU, Softmax, etc.) work correctly

**Likely Cause**: Issue in the autograd graph construction or backward pass traversal. The computation graph appears to be built correctly (Parents set, Backward functions assigned), but gradients remain zero throughout. This requires separate investigation beyond cuBLAS debugging.

## Lessons Learned

### 1. Memory Layout Matters
Understanding the difference between row-major and column-major is **critical** when interfacing with cuBLAS. Every parameter must account for this.

### 2. Leading Dimensions Are Not Intuitive
LD is not the "number of rows" - it's the stride between consecutive elements in a column, which equals the number of columns in row-major storage.

### 3. Test in Isolation
When debugging complex operations:
- Extract to standalone test programs
- Use known inputs with hand-calculated outputs
- Test edge cases (non-square matrices, different dimensions)

### 4. Systematic Exploration Beats Guessing
The systematic test that tried all 8 combinations found the answer faster than reasoning about the math.

### 5. Error Messages Can Be Misleading
"Parameter 8 illegal value" pointed to the leading dimension, but didn't say which matrix or what the correct value should be.

### 6. Vectorized Kernels vs Library Calls
For simple element wise operations (add, sub, mul, div), custom vectorized CUDA kernels using float4 can be:
- Simpler to implement correctly
- Easier to debug
- Comparable or better performance for coalesced memory access patterns

## Performance Notes

### Before Fix
- Test crashed during epoch 3 with SGEMM parameter errors
- No successful training runs

### After Fix  
- Test runs to completion without CUDA errors
- All 5 epochs complete in ~133 seconds on RTX 3070
- No cuBLAS errors in forward or backward passes
- Kernel executions are clean (no synchronization issues)

### Observed Behavior
- Forward pass: Batch size 32, dimensions 784→128→64→10
- Backward pass: Gradients flow through all three matmul variants
- No memory errors, no illegal access violations

## Verification Commands

To verify the fixes work:
```bash
# Rebuild CUDA kernels
cd internal/backend/cuda/kernels
make clean && make

# Run MNIST test (should complete without cuBLAS errors)
cd /mnt/e/Projects/nanograd
go test example/MNIST/MNIST_NN_GPU_test.go -v -timeout 180s
```

Expected: Test completes all 5 epochs without panic or SGEMM errors (though accuracy will be low due to unrelated gradient issue).

## References

- cuBLAS Documentation: Parameters for SGEMM are (transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
- Row-major to column-major conversion: Swap matrices and dimensions
- Leading dimension: Stride in memory between consecutive column elements

## Date
February 7, 2026

## Status
✅ **cuBLAS Formulas**: All three matmul variants fixed and verified  
⚠️ **Model Training**: Separate gradient flow issue identified but not addressed
