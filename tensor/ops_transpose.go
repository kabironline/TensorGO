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
		data:         t.data, // view: shares the parent's storage
		Shape:        newShape,
		Strides:      newStrides,
		grad:         nil,
		Parents:      []*Tensor{t},
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
	}

	out.Backward = func() {
		if out.grad == nil {
			return
		}

		n := TotalSize(t.Shape)
		parentGrad := t.Device.Allocate(n)
		t.Device.Fill(parentGrad, 0, n) // pool blocks are not zeroed
		defer t.Device.Free(parentGrad)

		og := out.Grad()

		// t's LOGICAL strides -- not t.Strides, which describe the storage layout.
		tStrides := ComputeStrides(t.Shape)
		coords := make([]int, len(out.Shape))

		for i := range og {
			// Decompose i into coordinates of out.
			idx := i
			for k := len(coords) - 1; k >= 0; k-- {
				coords[k] = idx % out.Shape[k]
				idx /= out.Shape[k]
			}
			// Axis k of out is axis order[k] of t, so the same coordinate contributes
			// through t's logical stride for that axis.
			tIdx := 0
			for k := range coords {
				tIdx += coords[k] * tStrides[order[k]]
			}
			parentGrad[tIdx] += og[i]
		}

		t.AccumulateGrad(parentGrad)
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

	src := t
	if !t.IsContiguous() {
		src = Contiguous(t)
	}

	out := &Tensor{
		data:         src.data, // shares t's storage, or the materialised copy's
		Shape:        actualShape,
		Strides:      ComputeStrides(actualShape),
		grad:         nil,
		Parents:      []*Tensor{t},
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
	}

	if t.RequiresGrad {
		out.Backward = func() {
			if out.grad == nil {
				return
			}
			t.AccumulateGrad(out.Grad()) // maps logical -> physical for t's strides
		}
	}

	return out
}
