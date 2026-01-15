package cpu

import "math"

// Sum computes the sum of all elements
func (bk *CPUBackend) Sum(a []float32, size int) float32 {
	var sum float32
	for i := range size {
		sum += a[i]
	}
	return sum
}

// Mean computes the mean of all elements
func (bk *CPUBackend) Mean(a []float32, size int) float32 {
	return bk.Sum(a, size) / float32(size)
}

// Max finds the maximum element
func (bk *CPUBackend) Max(a []float32, size int) float32 {
	maxVal := a[0]
	for i := 1; i < size; i++ {
		if a[i] > maxVal {
			maxVal = a[i]
		}
	}
	return maxVal
}

// Min finds the minimum element
func (bk *CPUBackend) Min(a []float32, size int) float32 {
	minVal := a[0]
	for i := 1; i < size; i++ {
		if a[i] < minVal {
			minVal = a[i]
		}
	}
	return minVal
}

// SumAxis computes sum along specified axis
func (bk *CPUBackend) SumAxis(a []float32, shape []int, axis int) []float32 {
	outShape := make([]int, len(shape))
	copy(outShape, shape)
	outShape[axis] = 1

	size := 1
	for _, s := range outShape {
		size *= s
	}
	out := bk.Allocate(size)

	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}

	outStrides := make([]int, len(outShape))
	s = 1
	for i := len(outShape) - 1; i >= 0; i-- {
		outStrides[i] = s
		s *= outShape[i]
	}

	coords := make([]int, len(shape))
	for i, val := range a {
		tempIdx := i
		for k := 0; k < len(shape); k++ {
			coords[k] = (tempIdx / strides[k]) % shape[k]
		}
		coords[axis] = 0

		outIdx := 0
		for k := 0; k < len(coords); k++ {
			outIdx += coords[k] * outStrides[k]
		}
		out[outIdx] += val
	}
	return out
}

// MaxAxis computes max along specified axis
func (bk *CPUBackend) MaxAxis(a []float32, shape []int, axis int) []float32 {
	// Simple implementation, similar to SumAxis but with Max
	outShape := make([]int, len(shape))
	copy(outShape, shape)
	outShape[axis] = 1

	size := 1
	for _, s := range outShape {
		size *= s
	}
	out := bk.Allocate(size)
	// Initialize with very small value
	for i := range out {
		out[i] = float32(math.Inf(-1))
	}

	strides := bk.computeStrides(shape)
	outStrides := bk.computeStrides(outShape)

	coords := make([]int, len(shape))
	for i, val := range a {
		tempIdx := i
		for k := 0; k < len(shape); k++ {
			coords[k] = (tempIdx / strides[k]) % shape[k]
		}
		coords[axis] = 0

		outIdx := 0
		for k := 0; k < len(coords); k++ {
			outIdx += coords[k] * outStrides[k]
		}
		if val > out[outIdx] {
			out[outIdx] = val
		}
	}
	return out
}

// MeanAxis computes mean along specified axis
func (bk *CPUBackend) MeanAxis(a []float32, shape []int, axis int) []float32 {
	sum := bk.SumAxis(a, shape, axis)
	count := float32(shape[axis])
	for i := range sum {
		sum[i] /= count
	}
	return sum
}

func (bk *CPUBackend) computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return strides
}
