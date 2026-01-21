package cpu

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// Matrix operations for the CPU backend.

func matMul(a, b []float32, m, n, k, sA, sB int, aT, bT bool, out []float32) []float32 {
	var aTransposed, bTransposed blas.Transpose = blas.NoTrans, blas.NoTrans
	rowsA, colsA := m, k
	rowsB, colsB := k, n

	if aT {
		aTransposed = blas.Trans
		rowsA, colsA = k, m
	}
	if bT {
		bTransposed = blas.Trans
		rowsB, colsB = n, k
	}

	blas32.Gemm(
		aTransposed, bTransposed,
		1.0, // alpha
		blas32.General{
			Rows:   rowsA,
			Cols:   colsA,
			Stride: sA,
			Data:   a,
		}, // A
		blas32.General{
			Rows:   rowsB,
			Cols:   colsB,
			Stride: sB,
			Data:   b,
		}, // B
		0.0, // beta
		blas32.General{
			Rows:   m,
			Cols:   n,
			Stride: n,
			Data:   out,
		}, // C
	)
	return out
}

// MatMul performs matrix multiplication: C = A @ B
// a: data buffer for matrix A with shape [m, k]
// b: data buffer for matrix B with shape [k, n]
// m, n, k: matrix dimensions
func (bk *CPUBackend) MatMul(a, b, out []float32, m, n, k, sA, sB int) []float32 {
	return matMul(a, b, m, n, k, sA, sB, false, false, out)
}

// MatMulTransA performs matrix multiplication with A transposed: C = A^T @ B
func (bk *CPUBackend) MatMulTransA(a, b, out []float32, m, n, k, sA, sB int) []float32 {
	return matMul(a, b, m, n, k, sA, sB, true, false, out)
}

// MatMulTransB performs matrix multiplication with B transposed: C = A @ B^T
func (bk *CPUBackend) MatMulTransB(a, b, out []float32, m, n, k, sA, sB int) []float32 {
	return matMul(a, b, m, n, k, sA, sB, false, true, out)
}

// MatMulAdd performs matrix multiplication and addition: C = A @ B + C
func (bk *CPUBackend) MatMulAdd(a, b, c, out []float32, m, n, k, sA, sB int) {
	// Perform C = A @ B + C
	// Gemm is alpha * A * B + beta * C
	// Here, alpha = 1.0, beta = 1.0

	copy(out, c) // Initialize out with current values of C

	blas32.Gemm(
		blas.NoTrans, blas.NoTrans,
		1.0, // alpha
		blas32.General{
			Rows:   m,
			Cols:   k,
			Stride: sA,
			Data:   a,
		}, // A
		blas32.General{
			Rows:   k,
			Cols:   n,
			Stride: sB,
			Data:   b,
		}, // B
		1.0, // beta
		blas32.General{
			Rows:   m,
			Cols:   n,
			Stride: n,
			Data:   out,
		}, // C
	)
}
