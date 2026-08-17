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
	idx := 0
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
	return t.Data()[t.getIndex(indices...)]
}

// SetAt sets the value at the given multi-dimensional indices.
func (t *Tensor) SetAt(val float32, indices ...int) {
	t.Data()[t.getIndex(indices...)] = val
}

// PhysicalIndexFromLinearIndex converts a logical flat index to the physical
// index in the underlying data buffer, accounting for strides and offset.
func (t *Tensor) PhysicalIndexFromLinearIndex(index int) int {
	if t.IsContiguous() {
		return index
	}
	physicalIndex := 0
	for i := len(t.Shape) - 1; i >= 0; i-- {
		physicalIndex += (index % t.Shape[i]) * t.Strides[i]
		index /= t.Shape[i]
	}
	return physicalIndex
}

// Contiguous returns a tensor with the same logical value as t, laid out
// contiguously in row-major order. If t is already contiguous it is returned
// unchanged; otherwise the data is materialised into a fresh buffer.
//
// The result stays connected to the autograd graph: a copy is the identity, so
// gradients flow straight back to t. It previously returned a detached tensor,
// which made `tensor.Contiguous(x)` a silent gradient sink -- and cost real
// debugging time three separate times.
func Contiguous(t *Tensor) *Tensor {
	if t.IsContiguous() {
		return t
	}
	return t.Clone()
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
