package tensor

import "math"

func (t *Tensor) ReLU() *Tensor {
	tContig := Contiguous(t)
	result := make([]float64, len(tContig.Data))
	for i, v := range tContig.Data {
		result[i] = max(0, v)
	}
	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := make([]float64, len(out.Grad))
		for i, v := range out.Data {
			if v > 0 {
				// Gradient flows only where ReLU is active
				grad[i] = out.Grad[i]
			}
		}
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Sigmoid() *Tensor {
	tContig := Contiguous(t)
	result := make([]float64, len(tContig.Data))
	for i, v := range tContig.Data {
		result[i] = 1 / (1 + math.Exp(-v))
	}

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := make([]float64, len(out.Grad))
		for i, v := range out.Data {
			// Gradient of sigmoid: s * (1 - s)
			grad[i] = out.Grad[i] * (v * (1 - v))
		}
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Tanh() *Tensor {
	tContig := Contiguous(t)
	result := make([]float64, len(tContig.Data))
	for i, v := range tContig.Data {
		result[i] = math.Tanh(v)
	}
	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := make([]float64, len(out.Grad))
		for i, v := range out.Data {
			// Gradient of tanh: 1 - tanh^2(x)
			grad[i] = out.Grad[i] * (1 - v*v)
		}
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Softmax() *Tensor {
	if len(t.Shape) > 2 {
		panic("Softmax only supports 1D or 2D tensors")
	}

	tContig := Contiguous(t)

	rows := 1
	cols := len(tContig.Data)
	if len(t.Shape) == 2 {
		rows = t.Shape[0]
		cols = t.Shape[1]
	}

	result := make([]float64, len(tContig.Data))

	for r := 0; r < rows; r++ {
		offset := r * cols
		// Find max for numerical stability
		maxVal := tContig.Data[offset]
		for c := 1; c < cols; c++ {
			if tContig.Data[offset+c] > maxVal {
				maxVal = tContig.Data[offset+c]
			}
		}

		// Compute exp and sum
		expSum := 0.0
		for c := 0; c < cols; c++ {
			ex := math.Exp(tContig.Data[offset+c] - maxVal)
			result[offset+c] = ex
			expSum += ex
		}

		// Normalize
		for c := 0; c < cols; c++ {
			result[offset+c] /= expSum
		}
	}

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		gradInput := make([]float64, len(out.Grad))

		for r := 0; r < rows; r++ {
			offset := r * cols
			// Compute sum(y_k * grad_k) for this row
			dot := 0.0
			for c := 0; c < cols; c++ {
				dot += out.Data[offset+c] * out.Grad[offset+c]
			}

			// Update gradients: y_i * (grad_i - dot)
			for c := 0; c < cols; c++ {
				y := out.Data[offset+c]
				grad := out.Grad[offset+c]
				gradInput[offset+c] = y * (grad - dot)
			}
		}
		t.AccumulateGrad(gradInput)
	}
	return out
}
