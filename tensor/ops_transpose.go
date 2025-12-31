package tensor

// Transpose returns a new tensor with its axes permuted according to the given order.
// For example, if the input tensor has shape (2, 3, 4) and the order is [1, 0, 2],
// the resulting tensor will have shape (3, 2, 4).
// It panics if the order is not a valid permutation of the tensor's dimensions.
func (t *Tensor) Transpose(order []int) *Tensor {
	// Using strides to update the values without copying data around
	// just changing the shape and strides accordingly
	if len(order) != len(t.Shape) {
		panic("Transpose: order length must match tensor dimensions")
	}

	newShape := make([]int, len(t.Shape))
	newStrides := make([]int, len(t.Strides))

	seen := make([]bool, len(t.Shape))
	for i, axis := range order {
		if axis < 0 || axis >= len(t.Shape) {
			panic("Transpose: invalid axis in order")

		}
		if seen[axis] {
			panic("Transpose: duplicate axis in order")
		}
		seen[axis] = true
		newShape[i] = t.Shape[axis]
		newStrides[i] = t.Strides[axis]
	}

	return &Tensor{
		Data:    t.Data,
		Shape:   newShape,
		Strides: newStrides,
		Offset:  t.Offset,
		Grad:    t.Grad,
		Parents: []*Tensor{t},
	}
}
