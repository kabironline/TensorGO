package cpu

import "gonum.org/v1/gonum/blas/blas64"

// DMAS operations for CPU backend.

// Add performs element-wise addition: out = a + b
func (bk *CPUBackend) Add(a, b []float64, size int) []float64 {
	out := bk.Allocate(size)
	copy(out, a)
	blas64.Axpy(
		1.0,
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: b,
		},
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}

// Sub performs element-wise subtraction: out = a - b
func (bk *CPUBackend) Sub(a, b []float64, size int) []float64 {
	out := bk.Allocate(size)
	copy(out, a)
	blas64.Axpy(
		-1.0,
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: b,
		},
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}

// Mul performs element-wise multiplication: out = a * b
func (bk *CPUBackend) Mul(a, b []float64, size int) []float64 {
	out := bk.Allocate(size)
	copy(out, a)
	for i := range out {
		out[i] *= b[i]
	}
	return out
}

// Div performs element-wise division: out = a / b
func (bk *CPUBackend) Div(a, b []float64, size int) []float64 {
	out := bk.Allocate(size)
	for i := range out {
		out[i] = a[i] / b[i]
	}
	return out
}

// Neg performs element-wise negation: out = -a
func (bk *CPUBackend) Neg(a []float64, size int) []float64 {
	out := bk.Allocate(size)
	for i := range out {
		out[i] = -a[i]
	}
	return out
}

// -------------------- Scalar Operations --------------------

// AddScalar performs element-wise scalar addition: out = a + scalar
func (bk *CPUBackend) AddScalar(a []float64, scalar float64, size int) []float64 {
	out := bk.Allocate(size)
	copy(out, a)
	// Use blas64.Axpy to add scalar*ones to out
	// Create a temporary slice of ones
	ones := make([]float64, size)
	for i := range ones {
		ones[i] = 1.0
	}
	blas64.Axpy(
		scalar,
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: ones,
		},
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}

// SubScalar performs element-wise scalar subtraction: out = a - scalar
func (bk *CPUBackend) SubScalar(a []float64, scalar float64, size int) []float64 {
	out := bk.Allocate(size)
	copy(out, a)
	// Use blas64.Axpy to subtract scalar*ones from out
	ones := make([]float64, size)
	for i := range ones {
		ones[i] = 1.0
	}
	blas64.Axpy(
		-scalar,
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: ones,
		},
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}

// MulScalar performs element-wise scalar multiplication: out = a * scalar
func (bk *CPUBackend) MulScalar(a []float64, scalar float64, size int) []float64 {
	out := bk.Allocate(size)
	copy(out, a)
	blas64.Scal(
		scalar,
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}

// DivScalar performs element-wise scalar division: out = a / scalar
func (bk *CPUBackend) DivScalar(a []float64, scalar float64, size int) []float64 {
	out := bk.Allocate(size)
	invScalar := 1.0 / scalar
	copy(out, a)
	blas64.Scal(
		invScalar,
		blas64.Vector{
			N:    size,
			Inc:  1,
			Data: out,
		})
	return out
}
