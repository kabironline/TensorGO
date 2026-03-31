package tensor

import "github.com/kabironline/nanograd/backend"

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
		// For GPU: copy gradient to CPU to read the scalar value
		var gradOut float32
		if out.Device.IsGPU() {
			if memTransfer, ok := out.Device.(backend.MemoryTransfer); ok {
				gradCPU := memTransfer.ToCPU(out.Grad)
				gradOut = gradCPU[0]
			}
		} else {
			gradOut = out.Grad[0]
		}

		// Create a gradient tensor full of gradOut using backend Fill operation
		grad := t.Device.Allocate(TotalSize(t.Shape))
		t.Device.Fill(grad, gradOut, TotalSize(t.Shape))

		t.AccumulateGrad(grad)
		t.Device.Free(grad)
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
		// For GPU: copy gradient to CPU to read the scalar value
		var gradOut float32
		if out.Device.IsGPU() {
			if memTransfer, ok := out.Device.(backend.MemoryTransfer); ok {
				gradCPU := memTransfer.ToCPU(out.Grad)
				gradOut = gradCPU[0]
			}
		} else {
			gradOut = out.Grad[0]
		}

		gradPerElement := gradOut / float32(TotalSize(t.Shape))

		// Create a gradient tensor using backend Fill operation
		grad := t.Device.Allocate(TotalSize(t.Shape))
		t.Device.Fill(grad, gradPerElement, TotalSize(t.Shape))

		t.AccumulateGrad(grad)
		t.Device.Free(grad)
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
