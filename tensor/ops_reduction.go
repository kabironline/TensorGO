package tensor

// Sum returns a new Tensor which is the sum of all elements in the original tensor.
func (t *Tensor) Sum() *Tensor {
	total := 0.0
	// Use Contiguous to handle views correctly
	tContig := Contiguous(t)
	for _, v := range tContig.Data {
		total += v
	}

	out := NewTensor([]float64{total}, []int{1}, t)
	// --- Backward function ---
	// During backpropagation, the gradient from the output (a single value)
	// needs to be distributed back to all elements of the original tensor.
	out.Backward = func() {
		gradOut := out.Grad[0] // Gradient from the output tensor

		// Create a gradient tensor full of gradOut
		grad := make([]float64, TotalSize(t.Shape))
		for i := range grad {
			grad[i] = gradOut
		}
		t.AccumulateGrad(grad)
	}
	return out
}

// Mean returns a new Tensor which is the mean of all elements in the original tensor.
func (t *Tensor) Mean() *Tensor {
	total := 0.0
	tContig := Contiguous(t)
	for _, v := range tContig.Data {
		total += v
	}
	mean := total / float64(len(tContig.Data))
	out := NewTensor([]float64{mean}, []int{1}, t)
	// --- Backward function ---
	// During backpropagation, the gradient from the output (a single value)
	// needs to be distributed back to all elements of the original tensor.
	out.Backward = func() {
		gradOut := out.Grad[0] // Gradient from the output tensor
		gradPerElement := gradOut / float64(TotalSize(t.Shape))

		grad := make([]float64, TotalSize(t.Shape))
		for i := range grad {
			grad[i] = gradPerElement
		}
		t.AccumulateGrad(grad)
	}
	return out
}

// ReduceSumTo matches the gradient shape of 'out' to the shape of 'parent'
// by summing across broadcasted dimensions.
func ReduceSumTo(grad []float64, gradShape, parentShape []int) []float64 {
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
		currentGrad = sumAcrossDimension(currentGrad, currentShape, 0)
		currentShape = currentShape[1:]
	}

	// 2. Sum across dimensions that were size 1 in parent but >1 in output
	// e.g., grad [32, 10], parent [1, 10] -> sum across dim 0
	for i := range parentShape {
		if parentShape[i] == 1 && currentShape[i] > 1 {
			currentGrad = sumAcrossDimension(currentGrad, currentShape, i)
			currentShape[i] = 1
		}
	}
	return currentGrad
}

// sumAcrossDimension sums a flattened tensor data along a specific dimension.
// It returns the new data and assumes the resulting shape has size 1 at that dimension.
func sumAcrossDimension(data []float64, shape []int, dim int) []float64 {
	// Calculate the size of the output (same rank, but dim is 1)
	outShape := make([]int, len(shape))
	copy(outShape, shape)
	outShape[dim] = 1

	outSize := TotalSize(outShape)
	outData := make([]float64, outSize)

	strides := defaultStrides(shape)
	outStrides := defaultStrides(outShape)

	// Iterate over the input data
	for i, val := range data {
		// Convert linear index to coordinates
		coords := CoordsFromLinearIndex(i, shape, strides)

		// Project coordinates to output (dim becomes 0)
		coords[dim] = 0

		// Convert back to linear index in output
		outIdx := LinearIndexFromCoords(coords, outShape, outStrides)

		outData[outIdx] += val
	}
	return outData
}
