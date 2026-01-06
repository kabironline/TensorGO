package cpu

import (
	"gonum.org/v1/gonum/floats"
)

// getTotalSize computes the total number of elements in a shape.
func getTotalSize(shape []int) int {
	total := 1
	for _, s := range shape {
		total *= s
	}
	return total
}

// getStrides computes row-major strides for a given shape.
func getStrides(shape []int) []int {
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return strides
}

// BroadcastAdd performs broadcasted element-wise addition using optimized BLAS where possible.
// For contiguous same-shape tensors, uses Gonum's SIMD-optimized AddTo.
// For different shapes, uses stride-aware iteration.
func (b *CPUBackend) BroadcastAdd(a, ai []float64, aShape, bShape, outShape []int) []float64 {
	outSize := getTotalSize(outShape)
	out := make([]float64, outSize)

	// Fast path: identical shapes, both contiguous, offset 0
	if len(aShape) == len(bShape) && shapesEqual(aShape, bShape) {
		floats.AddTo(out, a, ai)
		return out
	}

	// Slow path: use stride-aware broadcasting
	aStrides := getStrides(aShape)
	bStrides := getStrides(bShape)
	rank := len(outShape)

	coords := make([]int, rank)
	for i := 0; i < outSize; i++ {
		// Compute indices in a and b considering broadcasting
		aIdx := 0
		bIdx := 0
		shift := rank - len(aShape)
		for j, c := range coords {
			if j < shift {
				// Broadcast dimension (prepend 1s)
				continue
			}
			origJ := j - shift
			// If dimension is 1, stride is 0 (broadcasting)
			if aShape[origJ] == 1 {
				// aIdx += 0
			} else {
				aIdx += c * aStrides[origJ]
			}
		}

		bShift := rank - len(bShape)
		for j, c := range coords {
			if j < bShift {
				continue
			}
			origJ := j - bShift
			if bShape[origJ] == 1 {
				// bIdx += 0
			} else {
				bIdx += c * bStrides[origJ]
			}
		}

		out[i] = a[aIdx] + ai[bIdx]

		// Increment coordinates
		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < outShape[j] {
				break
			}
			coords[j] = 0
		}
	}

	return out
}

// BroadcastSub performs broadcasted element-wise subtraction.
func (b *CPUBackend) BroadcastSub(a, ai []float64, aShape, bShape, outShape []int) []float64 {
	outSize := getTotalSize(outShape)
	out := make([]float64, outSize)

	// Fast path: identical shapes, both contiguous
	if len(aShape) == len(bShape) && shapesEqual(aShape, bShape) {
		for i := range out {
			out[i] = a[i] - ai[i]
		}
		return out
	}

	// Slow path: stride-aware broadcasting
	aStrides := getStrides(aShape)
	bStrides := getStrides(bShape)
	rank := len(outShape)

	coords := make([]int, rank)
	for i := 0; i < outSize; i++ {
		aIdx := 0
		bIdx := 0
		aShift := rank - len(aShape)
		for j, c := range coords {
			if j < aShift {
				continue
			}
			origJ := j - aShift
			if aShape[origJ] != 1 {
				aIdx += c * aStrides[origJ]
			}
		}

		bShift := rank - len(bShape)
		for j, c := range coords {
			if j < bShift {
				continue
			}
			origJ := j - bShift
			if bShape[origJ] != 1 {
				bIdx += c * bStrides[origJ]
			}
		}

		out[i] = a[aIdx] - ai[bIdx]

		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < outShape[j] {
				break
			}
			coords[j] = 0
		}
	}

	return out
}

// BroadcastMul performs broadcasted element-wise multiplication.
func (b *CPUBackend) BroadcastMul(a, ai []float64, aShape, bShape, outShape []int) []float64 {
	outSize := getTotalSize(outShape)
	out := make([]float64, outSize)

	// Fast path: identical shapes, both contiguous
	if len(aShape) == len(bShape) && shapesEqual(aShape, bShape) {
		for i := range out {
			out[i] = a[i] * ai[i]
		}
		return out
	}

	// Slow path: stride-aware broadcasting
	aStrides := getStrides(aShape)
	bStrides := getStrides(bShape)
	rank := len(outShape)

	coords := make([]int, rank)
	for i := 0; i < outSize; i++ {
		aIdx := 0
		bIdx := 0
		aShift := rank - len(aShape)
		for j, c := range coords {
			if j < aShift {
				continue
			}
			origJ := j - aShift
			if aShape[origJ] != 1 {
				aIdx += c * aStrides[origJ]
			}
		}

		bShift := rank - len(bShape)
		for j, c := range coords {
			if j < bShift {
				continue
			}
			origJ := j - bShift
			if bShape[origJ] != 1 {
				bIdx += c * bStrides[origJ]
			}
		}

		out[i] = a[aIdx] * ai[bIdx]

		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < outShape[j] {
				break
			}
			coords[j] = 0
		}
	}

	return out
}

// BroadcastDiv performs broadcasted element-wise division.
func (b *CPUBackend) BroadcastDiv(a, ai []float64, aShape, bShape, outShape []int) []float64 {
	outSize := getTotalSize(outShape)
	out := make([]float64, outSize)

	// Fast path: identical shapes, both contiguous
	if len(aShape) == len(bShape) && shapesEqual(aShape, bShape) {
		for i := range out {
			out[i] = a[i] / ai[i]
		}
		return out
	}

	// Slow path: stride-aware broadcasting
	aStrides := getStrides(aShape)
	bStrides := getStrides(bShape)
	rank := len(outShape)

	coords := make([]int, rank)
	for i := 0; i < outSize; i++ {
		aIdx := 0
		bIdx := 0
		aShift := rank - len(aShape)
		for j, c := range coords {
			if j < aShift {
				continue
			}
			origJ := j - aShift
			if aShape[origJ] != 1 {
				aIdx += c * aStrides[origJ]
			}
		}

		bShift := rank - len(bShape)
		for j, c := range coords {
			if j < bShift {
				continue
			}
			origJ := j - bShift
			if bShape[origJ] != 1 {
				bIdx += c * bStrides[origJ]
			}
		}

		out[i] = a[aIdx] / ai[bIdx]

		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < outShape[j] {
				break
			}
			coords[j] = 0
		}
	}

	return out
}

// Helper to check if shapes are equal
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
