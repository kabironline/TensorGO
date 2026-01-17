package cpu

import (
	"gonum.org/v1/gonum/blas/blas32"
)

// DMAS operations for CPU backend.

// Add performs element-wise addition: out = a + b
func (bk *CPUBackend) Add(a, b, out []float32, size int) {
	copy(out, a)
	blas32.Axpy(
		1.0,
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: b,
		},
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
}

// Sub performs element-wise subtraction: out = a - b
func (bk *CPUBackend) Sub(a, b, out []float32, size int) {
	copy(out, a)
	blas32.Axpy(
		-1.0,
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: b,
		},
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
}

// Mul performs element-wise multiplication: out = a * b
func (bk *CPUBackend) Mul(a, b, out []float32, size int) {
	copy(out, a)
	for i := range out {
		out[i] *= b[i]
	}
}

// Div performs element-wise division: out = a / b
func (bk *CPUBackend) Div(a, b, out []float32, size int) {
	for i := range out {
		out[i] = a[i] / b[i]
	}
}

// Neg performs element-wise negation: out = -a
func (bk *CPUBackend) Neg(a, out []float32, size int) {
	for i := range out {
		out[i] = -a[i]
	}
}

// -------------------- Scalar Operations --------------------

// AddScalar performs element-wise scalar addition: out = a + scalar
func (bk *CPUBackend) AddScalar(a []float32, scalar float32, size int) []float32 {
	out := bk.Allocate(size)
	copy(out, a)
	// Use blas32.Axpy to add scalar*ones to out
	// Create a temporary slice of ones
	ones := make([]float32, size)
	for i := range ones {
		ones[i] = 1.0
	}
	blas32.Axpy(
		scalar,
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: ones,
		},
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}

// SubScalar performs element-wise scalar subtraction: out = a - scalar
func (bk *CPUBackend) SubScalar(a []float32, scalar float32, size int) []float32 {
	out := bk.Allocate(size)
	copy(out, a)
	// Use blas32.Axpy to subtract scalar*ones from out
	ones := make([]float32, size)
	for i := range ones {
		ones[i] = 1.0
	}
	blas32.Axpy(
		-scalar,
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: ones,
		},
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}

// MulScalar performs element-wise scalar multiplication: out = a * scalar
func (bk *CPUBackend) MulScalar(a []float32, scalar float32, size int) []float32 {
	out := bk.Allocate(size)
	copy(out, a)
	blas32.Scal(
		scalar,
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}

// DivScalar performs element-wise scalar division: out = a / scalar
func (bk *CPUBackend) DivScalar(a []float32, scalar float32, size int) []float32 {
	out := bk.Allocate(size)
	invScalar := 1.0 / scalar
	copy(out, a)
	blas32.Scal(
		invScalar,
		blas32.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}
