package tensor

import (
	"gonum.org/v1/gonum/floats"
)

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

	for i, _ := range targetShape {
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
		Data:    t.Data,
		Shape:   append([]int(nil), targetShape...),
		Strides: newStrides,
		Offset:  t.Offset,
		// Grad allocated lazily
		Grad:    nil,
		Parents: []*Tensor{t},
	}

	out.Backward = func() {
		gradReduced := ReduceSumTo(out.Grad, out.Shape, t.Shape)
		t.AccumulateGrad(gradReduced)
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
// It uses Gonum's optimized floats.AddTo for contiguous cases and falls back to a stride-aware iterator.
func BroadcastAddOp(a, b *Tensor) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)
	total := TotalSize(outShape)
	outData := make([]float64, total)

	// Fast path: same shape and contiguous row-major
	if shapesEqual(a.Shape, b.Shape) && a.Contiguous() && b.Contiguous() && a.Offset == 0 && b.Offset == 0 {
		floats.AddTo(outData, a.Data, b.Data)
		return NewTensor(outData, outShape)
	}

	// Slow path: N-D iterator for broadcasting or non-contiguous views
	aB := a.BroadcastTo(outShape)
	bB := b.BroadcastTo(outShape)
	rank := len(outShape)
	if rank == 0 {
		outData[0] = aB.Data[aB.Offset] + bB.Data[bB.Offset]
		return NewTensor(outData, outShape)
	}

	coords := make([]int, rank)
	ai, bi := aB.Offset, bB.Offset
	for i := range total {
		outData[i] = aB.Data[ai] + bB.Data[bi]
		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < outShape[j] {
				ai += aB.Strides[j]
				bi += bB.Strides[j]
				break
			}
			ai -= (outShape[j] - 1) * aB.Strides[j]
			bi -= (outShape[j] - 1) * bB.Strides[j]
			coords[j] = 0
		}
	}
	return NewTensor(outData, outShape)
}

// BroadcastSubOp performs broadcasted element-wise subtraction.
func BroadcastSubOp(a, b *Tensor) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)
	total := TotalSize(outShape)
	outData := make([]float64, total)

	if shapesEqual(a.Shape, b.Shape) && a.Contiguous() && b.Contiguous() && a.Offset == 0 && b.Offset == 0 {
		floats.SubTo(outData, a.Data, b.Data)
		return NewTensor(outData, outShape)
	}

	aB := a.BroadcastTo(outShape)
	bB := b.BroadcastTo(outShape)
	rank := len(outShape)
	if rank == 0 {
		outData[0] = aB.Data[aB.Offset] - bB.Data[bB.Offset]
		return NewTensor(outData, outShape)
	}

	coords := make([]int, rank)
	ai, bi := aB.Offset, bB.Offset
	for i := range total {
		outData[i] = aB.Data[ai] - bB.Data[bi]
		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < outShape[j] {
				ai += aB.Strides[j]
				bi += bB.Strides[j]
				break
			}
			ai -= (outShape[j] - 1) * aB.Strides[j]
			bi -= (outShape[j] - 1) * bB.Strides[j]
			coords[j] = 0
		}
	}
	return NewTensor(outData, outShape)
}

// BroadcastMulOp performs broadcasted element-wise multiplication.
func BroadcastMulOp(a, b *Tensor) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)
	total := TotalSize(outShape)
	outData := make([]float64, total)

	if shapesEqual(a.Shape, b.Shape) && a.Contiguous() && b.Contiguous() && a.Offset == 0 && b.Offset == 0 {
		floats.MulTo(outData, a.Data, b.Data)
		return NewTensor(outData, outShape)
	}

	aB := a.BroadcastTo(outShape)
	bB := b.BroadcastTo(outShape)
	rank := len(outShape)
	if rank == 0 {
		outData[0] = aB.Data[aB.Offset] * bB.Data[bB.Offset]
		return NewTensor(outData, outShape)
	}

	coords := make([]int, rank)
	ai, bi := aB.Offset, bB.Offset
	for i := range total {
		outData[i] = aB.Data[ai] * bB.Data[bi]
		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < outShape[j] {
				ai += aB.Strides[j]
				bi += bB.Strides[j]
				break
			}
			ai -= (outShape[j] - 1) * aB.Strides[j]
			bi -= (outShape[j] - 1) * bB.Strides[j]
			coords[j] = 0
		}
	}
	return NewTensor(outData, outShape)
}

// BroadcastDivOp performs broadcasted element-wise division.
func BroadcastDivOp(a, b *Tensor) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)
	total := TotalSize(outShape)
	outData := make([]float64, total)

	if shapesEqual(a.Shape, b.Shape) && a.Contiguous() && b.Contiguous() && a.Offset == 0 && b.Offset == 0 {
		floats.DivTo(outData, a.Data, b.Data)
		return NewTensor(outData, outShape)
	}

	aB := a.BroadcastTo(outShape)
	bB := b.BroadcastTo(outShape)
	rank := len(outShape)
	if rank == 0 {
		outData[0] = aB.Data[aB.Offset] / bB.Data[bB.Offset]
		return NewTensor(outData, outShape)
	}

	coords := make([]int, rank)
	ai, bi := aB.Offset, bB.Offset
	for i := range total {
		outData[i] = aB.Data[ai] / bB.Data[bi]
		for j := rank - 1; j >= 0; j-- {
			coords[j]++
			if coords[j] < outShape[j] {
				ai += aB.Strides[j]
				bi += bB.Strides[j]
				break
			}
			ai -= (outShape[j] - 1) * aB.Strides[j]
			bi -= (outShape[j] - 1) * bB.Strides[j]
			coords[j] = 0
		}
	}
	return NewTensor(outData, outShape)
}
