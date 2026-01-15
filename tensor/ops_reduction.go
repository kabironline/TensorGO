package tensor

import (
	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/pools"
)

// Sum returns a new Tensor which is the sum of all elements in the original tensor.
func (t *Tensor) Sum() *Tensor {
	// Use Contiguous to handle views correctly
	tContig := Contiguous(t)
	total := t.Device.Sum(tContig.Data, len(tContig.Data))

	out := NewTensor([]float32{total}, []int{1}, t)
	// --- Backward function ---
	// During backpropagation, the gradient from the output (a single value)
	// needs to be distributed back to all elements of the original tensor.
	out.Backward = func() {
		gradOut := out.Grad[0] // Gradient from the output tensor

		// Create a gradient tensor full of gradOut
		grad := pools.GetBuffer(TotalSize(t.Shape))
		for i := range grad {
			grad[i] = gradOut
		}
		t.AccumulateGrad(grad)
		pools.PutBuffer(grad)
	}
	return out
}

// Mean returns a new Tensor which is the mean of all elements in the original tensor.
func (t *Tensor) Mean() *Tensor {
	tContig := Contiguous(t)
	mean := t.Device.Mean(tContig.Data, len(tContig.Data))

	out := NewTensor([]float32{mean}, []int{1}, t)
	// --- Backward function ---
	// During backpropagation, the gradient from the output (a single value)
	// needs to be distributed back to all elements of the original tensor.
	out.Backward = func() {
		gradOut := out.Grad[0] // Gradient from the output tensor
		gradPerElement := gradOut / float32(TotalSize(t.Shape))

		grad := pools.GetBuffer(TotalSize(t.Shape))
		for i := range grad {
			grad[i] = gradPerElement
		}
		t.AccumulateGrad(grad)
		pools.PutBuffer(grad)
	}
	return out
}

// ReduceSumTo matches the gradient shape of 'out' to the shape of 'parent'
// by summing across broadcasted dimensions.
func ReduceSumTo(dev backend.Backend, grad []float32, gradShape, parentShape []int) []float32 {
	// If shapes match exactly, no reduction needed
	if shapesEqual(gradShape, parentShape) {
		return grad
	}

	currentGrad := grad
	currentShape := make([]int, len(gradShape))
	copy(currentShape, gradShape)

	// 1. Sum across leading dimensions if grad has higher rank
	// e.g., grad [32, 10], parent [10] -> sum across dim 0
	for len(currentShape) > len(parentShape) {
		currentGrad = dev.SumAxis(currentGrad, currentShape, 0)
		currentShape = currentShape[1:]
	}

	// 2. Sum across dimensions that were size 1 in parent but >1 in output
	// e.g., grad [32, 10], parent [1, 10] -> sum across dim 0
	for i := range parentShape {
		if parentShape[i] == 1 && currentShape[i] > 1 {
			currentGrad = dev.SumAxis(currentGrad, currentShape, i)
			currentShape[i] = 1
		}
	}
	return currentGrad
}
