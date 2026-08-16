package tensor

// broadcastShapes determines the broadcasted shape of two tensors.
// It follows NumPy-style broadcasting rules:
// 1. If ranks differ, prepend 1s to the smaller rank shape.
// 2. For each dimension, sizes must either match or one must be 1.
func broadcastShapes(shapeA, shapeB []int) []int {
	maxLen := len(shapeA)
	if len(shapeB) > maxLen {
		maxLen = len(shapeB)
	}

	broadcastShape := make([]int, maxLen)
	for i := 0; i < maxLen; i++ {
		var dimA, dimB int
		if i < len(shapeA) {
			dimA = shapeA[len(shapeA)-1-i]
		} else {
			dimA = 1
		}
		if i < len(shapeB) {
			dimB = shapeB[len(shapeB)-1-i]
		} else {
			dimB = 1
		}

		if dimA != dimB && dimA != 1 && dimB != 1 {
			panic("broadcastShapes: shapes cannot be broadcasted")
		}

		broadcastShape[maxLen-1-i] = max(dimA, dimB)
	}
	return broadcastShape
}

// BroadcastTo returns a new tensor that is a broadcasted view of the original tensor
// to the specified shape. It does not copy the underlying data.
func (t *Tensor) BroadcastTo(targetShape []int) *Tensor {
	if len(targetShape) < len(t.Shape) {
		panic("BroadcastTo: target shape rank must be >= original rank")
	}

	newStrides := make([]int, len(targetShape))
	shift := len(targetShape) - len(t.Shape)

	for i := range targetShape {
		var origDim, origStride int
		if i < shift {
			origDim = 1
			origStride = 0
		} else {
			origDim = t.Shape[i-shift]
			origStride = t.Strides[i-shift]
		}

		if targetShape[i] != origDim && origDim != 1 {
			panic("BroadcastTo: incompatible dimensions")
		}

		if origDim == 1 {
			// Dimension of size 1 is broadcasted: stride is 0
			newStrides[i] = 0
		} else {
			newStrides[i] = origStride
		}
	}

	out := &Tensor{
		data:         t.data, // view: shares the parent's storage
		Shape:        append([]int(nil), targetShape...),
		Strides:      newStrides,
		grad:         nil,
		Parents:      []*Tensor{t},
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		contiguous:   false, // broadcasted tensors are not contiguous
	}

	out.Backward = func() {
		gradReduced := ReduceSumTo(t.Device, out.Grad(), out.Shape, t.Shape)
		t.AccumulateGrad(gradReduced)
		if !sameShape(out.Shape, t.Shape) {
			t.Device.Free(gradReduced)
		}
	}

	return out
}

// shapesEqual is a helper to check if two shapes are identical.
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

// BroadcastAddOp performs broadcasted element-wise addition.
func BroadcastAddOp(a, b *Tensor) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)

	// Use the backend's BroadcastAdd
	// Both inputs must be contiguous or the backend must handle views
	// For now, we ensure they are contiguous for the backend call
	aContig := Contiguous(a)
	bContig := Contiguous(b)

	outData := a.Device.BroadcastAdd(aContig.Data(), bContig.Data(), aContig.Shape, bContig.Shape, outShape)

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(outData),
		Shape:        outShape,
		Strides:      ComputeStrides(outShape),
		Device:       a.Device,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		Parents:      []*Tensor{a, b},
		contiguous:   true,
	}

	out.Backward = func() {
		// formula for gradient accumulation:
		// grad_a = ReduceSumTo(grad_out, out_shape, a_shape)
		// grad_b = ReduceSumTo(grad_out, out_shape, b_shape)
		if out.grad == nil {
			return
		}
		gradOut := out.Grad()

		if a.RequiresGrad {
			gradA := ReduceSumTo(a.Device, gradOut, out.Shape, a.Shape)
			a.AccumulateGrad(gradA)
			if !sameShape(out.Shape, a.Shape) {
				a.Device.Free(gradA)
			}
		}

		if b.RequiresGrad {
			gradB := ReduceSumTo(b.Device, gradOut, out.Shape, b.Shape)
			b.AccumulateGrad(gradB)
			if !sameShape(out.Shape, b.Shape) {
				b.Device.Free(gradB)
			}
		}
	}

	return out
}

// BroadcastSubOp performs broadcasted element-wise subtraction.
func BroadcastSubOp(a, b *Tensor) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)

	aContig := Contiguous(a)
	bContig := Contiguous(b)

	outData := a.Device.BroadcastSub(aContig.Data(), bContig.Data(), aContig.Shape, bContig.Shape, outShape)

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(outData),
		Shape:        outShape,
		Strides:      ComputeStrides(outShape),
		Device:       a.Device,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		Parents:      []*Tensor{a, b},
		contiguous:   true,
	}
	out.Backward = func() {
		// out = a - b, so:
		//   grad_a =  ReduceSumTo(grad_out, out_shape, a_shape)
		//   grad_b = -ReduceSumTo(grad_out, out_shape, b_shape)
		// The sign on b is what distinguishes this from BroadcastAddOp.
		if out.grad == nil {
			return
		}
		gradOut := out.Grad()

		if a.RequiresGrad {
			gradA := ReduceSumTo(a.Device, gradOut, out.Shape, a.Shape)
			a.AccumulateGrad(gradA)
			if !sameShape(out.Shape, a.Shape) {
				a.Device.Free(gradA)
			}
		}

		if b.RequiresGrad {
			negGrad := b.Device.MulScalar(gradOut, -1.0, len(gradOut))
			gradB := ReduceSumTo(b.Device, negGrad, out.Shape, b.Shape)
			b.AccumulateGrad(gradB)
			if !sameShape(out.Shape, b.Shape) {
				b.Device.Free(gradB)
			}
			b.Device.Free(negGrad)
		}
	}
	return out
}

// BroadcastMulOp performs broadcasted element-wise multiplication.
func BroadcastMulOp(a, b *Tensor) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)

	aContig := Contiguous(a)
	bContig := Contiguous(b)

	outData := a.Device.BroadcastMul(aContig.Data(), bContig.Data(), aContig.Shape, bContig.Shape, outShape)

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(outData),
		Shape:        outShape,
		Strides:      ComputeStrides(outShape),
		Device:       a.Device,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		Parents:      []*Tensor{a, b},
		contiguous:   true,
	}

	out.Backward = func() {
		// formula for gradient accumulation:
		// grad_a = ReduceSumTo(grad_out * b, out_shape, a_shape)
		// grad_b = ReduceSumTo(grad_out * a, out_shape, b_shape)
		if out.grad == nil {
			return
		}
		gradOut := out.Grad()

		if a.RequiresGrad {
			gradAData := a.Device.BroadcastMul(gradOut, bContig.Data(), out.Shape, bContig.Shape, out.Shape)
			gradA := ReduceSumTo(a.Device, gradAData, out.Shape, a.Shape)
			a.AccumulateGrad(gradA)
			if !sameShape(out.Shape, a.Shape) {
				a.Device.Free(gradA)
			}
		}

		if b.RequiresGrad {
			gradBData := a.Device.BroadcastMul(gradOut, aContig.Data(), out.Shape, aContig.Shape, out.Shape)
			gradB := ReduceSumTo(b.Device, gradBData, out.Shape, b.Shape)
			b.AccumulateGrad(gradB)
			if !sameShape(out.Shape, b.Shape) {
				b.Device.Free(gradB)
			}
		}
	}

	return out
}

// BroadcastDivOp performs broadcasted element-wise division.
func BroadcastDivOp(a, b *Tensor) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)

	aContig := Contiguous(a)
	bContig := Contiguous(b)

	outData := a.Device.BroadcastDiv(aContig.Data(), bContig.Data(), aContig.Shape, bContig.Shape, outShape)

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(outData),
		Shape:        outShape,
		Strides:      ComputeStrides(outShape),
		Device:       a.Device,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		Parents:      []*Tensor{a, b},
		contiguous:   true,
	}
	out.Backward = func() {
		// formula for gradient accumulation:
		// grad_a = ReduceSumTo(grad_out / b, out_shape, a_shape)
		// grad_b = ReduceSumTo(-grad_out * a / (b^2), out_shape, b_shape)
		gradOut := out.Grad()

		if a.RequiresGrad {
			gradA := ReduceSumTo(a.Device, gradOut, out.Shape, a.Shape)
			a.AccumulateGrad(gradA)
			if !sameShape(out.Shape, a.Shape) {
				a.Device.Free(gradA)
			}
		}

		if b.RequiresGrad {
			gradB := ReduceSumTo(b.Device, gradOut, out.Shape, b.Shape)
			b.AccumulateGrad(gradB)
			if !sameShape(out.Shape, b.Shape) {
				b.Device.Free(gradB)
			}
		}
	}
	return out
}
