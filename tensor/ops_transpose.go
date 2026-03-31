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

	out := &Tensor{
		Data:         t.Data,
		Shape:        newShape,
		Strides:      newStrides,
		Offset:       t.Offset,
		Grad:         nil,
		Parents:      []*Tensor{t},
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
	}

	out.Backward = func() {
		// Nothing to do if no gradient was produced for the output.
		if out.Grad == nil {
			return
		}

		// Ensure parent has a gradient buffer to accumulate into.
		if t.Grad == nil {
			t.AllocGrad()
		}

		// Build inverse permutation: for each axis k in the original tensor,
		// inv[k] gives the index in out.Shape corresponding to that axis.
		inv := make([]int, len(order))
		for i, axis := range order {
			inv[axis] = i
		}

		total := TotalSize(out.Shape)
		if total == 0 {
			return
		}

		// Fast-path: identity permutation -> direct accumulation.
		isIdentity := true
		for k := range order {
			if order[k] != k {
				isIdentity = false
				break
			}
		}
		if isIdentity {
			t.AccumulateGrad(out.Grad)
			return
		}

		// Generic path: iterate logical coordinates of the output once and map
		// them to the parent's physical indices.
		coords := make([]int, len(out.Shape))
		for i := 0; i < total; i++ {
			idx := i
			for k := len(coords) - 1; k >= 0; k-- {
				coords[k] = idx % out.Shape[k]
				idx /= out.Shape[k]
			}

			tIdx := t.Offset
			for k := 0; k < len(coords); k++ {
				tIdx += coords[inv[k]] * t.Strides[k]
			}
			t.Grad[tIdx] += out.Grad[i]
		}
	}

	return out
}

// Reshape returns a new tensor with the same data but a different shape.
func (t *Tensor) Reshape(newShape []int) *Tensor {
	totalOld := TotalSize(t.Shape)

	// Handle inferred dimension (-1)
	actualShape := make([]int, len(newShape))
	copy(actualShape, newShape)
	inferredIdx := -1
	totalKnown := 1
	for i, dim := range newShape {
		if dim == -1 {
			if inferredIdx != -1 {
				panic("Reshape: only one dimension can be inferred (-1)")
			}
			inferredIdx = i
		} else {
			totalKnown *= dim
		}
	}

	if inferredIdx != -1 {
		if totalOld%totalKnown != 0 {
			panic("Reshape: total size not divisible by known dimensions")
		}
		actualShape[inferredIdx] = totalOld / totalKnown
	}

	totalNew := TotalSize(actualShape)
	if totalOld != totalNew {
		panic("Reshape: total size must remain the same")
	}

	out := &Tensor{
		Data:    t.Data,
		Shape:   actualShape,
		Strides: ComputeStrides(actualShape),
		Offset:  t.Offset,
		// Grad allocated lazily
		Grad:         nil,
		Parents:      []*Tensor{t},
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
	}

	if t.RequiresGrad {
		out.Backward = func() {
			if out.Grad == nil {
				return
			}
			// Gradient of reshape is just reshape back to original shape
			// and accumulate. Since Data is shared, we just pass the buffer.
			t.AccumulateGrad(out.Grad)
		}
	}

	return out
}
