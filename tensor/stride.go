package tensor

import (
	"fmt"
)

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
func (t *Tensor) At(indices ...int) float32 {
	return t.Data[t.getIndex(indices...)]
}

// SetAt sets the value at the given multi-dimensional indices.
func (t *Tensor) SetAt(val float32, indices ...int) {
	t.Data[t.getIndex(indices...)] = val
}

// PhysicalIndexFromLinearIndex converts a logical flat index to the physical
// index in the underlying data buffer, accounting for strides and offset.
func (t *Tensor) PhysicalIndexFromLinearIndex(index int) int {
	if t.Contiguous() {
		return t.Offset + index
	}
	physicalIndex := t.Offset
	for i := len(t.Shape) - 1; i >= 0; i-- {
		physicalIndex += (index % t.Shape[i]) * t.Strides[i]
		index /= t.Shape[i]
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
	totalSize := TotalSize(shape)
	newData := t.Device.Allocate(totalSize)

	// Let the backend perform the contiguous copy (handles device-specific paths)
	// Provide the data starting at the tensor's offset so the backend's mapping
	// logic can operate as if the logical origin is index 0.
	// Backends implement Contiguous(dst) by writing into the provided out buffer.
	t.Device.Contiguous(t.Data, newData, shape, t.Strides, t.Offset)

	return &Tensor{
		Data:         newData,
		Shape:        append([]int{}, shape...),
		Strides:      ComputeStrides(shape),
		Grad:         nil,
		contiguous:   true,
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
	}
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
