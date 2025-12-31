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

// IsContiguous checks if the tensor's data is stored in a contiguous block of memory in row-major order.
func IsContiguous(shape, strides []int) bool {
	expectedStride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] == 1 {
			continue // Skip dimensions of size 1
		}
		if strides[i] != expectedStride {
			return false
		}
		expectedStride *= shape[i]
	}
	return true
}

// Contiguous returns a new tensor that is a contiguous copy of the original tensor.
// If the original tensor is already contiguous, it returns the original tensor.
func Contiguous(t *Tensor) *Tensor {
	if IsContiguous(t.Shape, t.Strides) {
		return t
	}

	totalSize := TotalSize(t.Shape)
	newData := make([]float64, totalSize)
	defStrides := defaultStrides(t.Shape)

	for i := range totalSize {
		coords := CoordsFromLinearIndex(i, t.Shape, defStrides)
		// Map logical coords to physical index in original (possibly non-contiguous) data
		oldIdx := LinearIndexFromCoords(coords, t.Shape, t.Strides) + t.Offset
		newData[i] = t.Data[oldIdx]
	}

	return &Tensor{
		Data:    newData,
		Shape:   append([]int{}, t.Shape...),
		Strides: defStrides,
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

// ValidateIndexes checks if the provided indexes are within the bounds of the tensor's shape.
// It panics if any index is out of bounds.
func ValidateIndexes(indexes []int, shape []int) {
	for i, idx := range indexes {
		if idx < 0 || idx >= shape[i] {
			panic("Index out of bounds")
		}
	}
}
