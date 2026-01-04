package tensor

// TotalSize calculates the total number of elements in a tensor given its shape.
func TotalSize(shape []int) int {
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	return total
}

// LinearIndexFromCoords converts multi-dimensional coordinates to a linear index based on the tensor's strides.
// Note: We do not perform bounds checking against TotalSize(shape) here because for views (like slices or broadcasts),
// the physical index can exceed the logical total size of the view.
func LinearIndexFromCoords(coords, shape, strides []int) int {
	if len(coords) != len(strides) {
		panic("LinearIndexFromCoords: Length of coords and strides must be the same")
	}
	index := 0
	for i := range coords {
		index += coords[i] * strides[i]
	}
	return index
}

// CoordsFromLinearIndex converts a flat logical index into coordinates for a given shape and its default strides.
func CoordsFromLinearIndex(index int, shape, strides []int) []int {
	coords := make([]int, len(shape))
	for i := range shape {
		coords[i] = (index / strides[i]) % shape[i]
	}
	return coords
}

// PhysicalIndexFromLinearIndex converts a flat logical index into the corresponding physical index in the tensor's data array,
// taking into account the tensor's strides and offset.
// This optimized version avoids allocating intermediate coordinate slices.
func PhysicalIndexFromLinearIndex(index int, shape, strides []int, offset int) int {
	physicalIndex := offset
	for i := len(shape) - 1; i >= 0; i-- {
		dim := shape[i]
		physicalIndex += (index % dim) * strides[i]
		index /= dim
	}
	return physicalIndex
}

// IsContiguous checks if the tensor's data is stored in a contiguous block of memory in row-major order.
func IsContiguous(shape, strides []int) bool {
	expected := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if s := shape[i]; s > 1 {
			if strides[i] != expected {
				return false
			}
			expected *= s
		} else if s == 0 {
			return true
		}
	}
	return true
}

// Contiguous returns a new tensor that is a contiguous copy of the original tensor.
// If the original tensor is already contiguous, it returns the original tensor.
func Contiguous(t *Tensor) *Tensor {
	if IsContiguous(t.Shape, t.Strides) {
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
		Data:    newData,
		Shape:   append([]int{}, shape...),
		Strides: computeStrides(shape),
		Grad:    make([]float64, totalSize),
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
