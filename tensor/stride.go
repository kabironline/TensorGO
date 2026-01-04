package tensor

import "fmt"

// TotalSize calculates the total number of elements in a tensor given its shape.
func TotalSize(shape []int) int {
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	return total
}

// getIndex converts multidimensional coordinates into a physical index into
// the underlying data slice, validating bounds and taking into account the
// tensor's offset and strides.
func (t *Tensor) getIndex(indices ...int) int {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("expected %d indices, got %d", len(t.Shape), len(indices)))
	}
	idx := t.Offset
	for i, ind := range indices {
		if ind < 0 || ind >= t.Shape[i] {
			panic(fmt.Sprintf("index %d out of bounds for dimension %d (size %d)", ind, i, t.Shape[i]))
		}
		idx += ind * t.Strides[i]
	}
	return idx
}

// At returns the element at the given multi-dimensional indices.
func (t *Tensor) At(indices ...int) float64 {
	return t.Data[t.getIndex(indices...)]
}

// SetAt sets the value at the given multi-dimensional indices.
func (t *Tensor) SetAt(val float64, indices ...int) {
	t.Data[t.getIndex(indices...)] = val
}

// PhysicalIndexFromLinearIndex converts a logical flat index to the physical
// index in the underlying data buffer, accounting for strides and offset.
func (t *Tensor) PhysicalIndexFromLinearIndex(index int) int {
	physicalIndex := t.Offset
	for i := len(t.Shape) - 1; i >= 0; i-- {
		dim := t.Shape[i]
		physicalIndex += (index % dim) * t.Strides[i]
		index /= dim
	}
	return physicalIndex
}

// Contiguous returns a new tensor that is a contiguous copy of the original tensor.
// If the original tensor is already contiguous, it returns the original tensor.
func Contiguous(t *Tensor) *Tensor {
	if t.Contiguous() {
		return t
	}

	shape := t.Shape
	strides := t.Strides
	rank := len(shape)
	totalSize := TotalSize(shape)

	newData := make([]float64, totalSize)
	tData := t.Data

	// Optimized iterator: avoids O(D) work and allocations inside the loop.
	// It uses an amortized O(1) coordinate update strategy.
	coords := make([]int, rank)
	currPos := t.Offset

	// Precompute backsteps to avoid multiplication during iteration.
	backsteps := make([]int, rank)
	for i := range shape {
		backsteps[i] = shape[i] * strides[i]
	}

	for i := range totalSize {
		newData[i] = tData[currPos]

		// Increment coordinates and update physical position
		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			currPos += strides[j]
			if coords[j] < shape[j] {
				break
			}
			// Dimension wrap-around
			currPos -= backsteps[j]
			coords[j] = 0
		}
	}

	return &Tensor{
		Data:       newData,
		Shape:      append([]int{}, shape...),
		Strides:    computeStrides(shape),
		Grad:       nil,
		contiguous: true,
	}
}

// NormalizeIndexes ensures that all indexes are non-negative by converting negative indexes to their positive equivalents.
func NormalizeIndexes(indexes []int, shape []int) []int {
	normalized := make([]int, len(indexes))
	for i, idx := range indexes {
		if idx < 0 {
			normalized[i] = shape[i] + idx
		} else {
			normalized[i] = idx
		}
	}
	return normalized
}

// ComputeStrides calculates the default row-major strides for a given shape.
func ComputeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	if len(shape) == 0 {
		return strides
	}
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// ValidateIndexes checks if the provided indexes are within the bounds of the tensor's shape.
// It panics if any index is out of bounds.
func ValidateIndexes(indexes []int, shape []int) {
	for i, idx := range indexes {
		if idx < 0 || idx >= shape[i] {
			panic("Index out of bounds")
		}
	}
}
