package tensor

import "math"

func (t *Tensor) ReLU() *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = max(0, v)
	}
	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		for i, v := range out.Data {
			if v > 0 {
				// Gradient flows only where ReLU is active
				t.Grad[i] += out.Grad[i]
			}
		}
	}
	return out
}

func (t *Tensor) Sigmoid() *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = 1 / (1 + math.Exp(-v))
	}

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		for i, v := range out.Data {
			// Gradient of sigmoid: s * (1 - s)
			t.Grad[i] += out.Grad[i] * (v * (1 - v))
		}
	}
	return out
}

func (t *Tensor) Tanh() *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = math.Tanh(v)
	}
	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		for i, v := range out.Data {
			// Gradient of tanh: 1 - tanh^2(x)
			t.Grad[i] += out.Grad[i] * (1 - v*v)
		}
	}
	return out
}

func (t *Tensor) Softmax() *Tensor {
	if len(t.Shape) > 2 {
		panic("Softmax only supports 1D or 2D tensors")
	}

	rows := 1
	cols := len(t.Data)
	if len(t.Shape) == 2 {
		rows = t.Shape[0]
		cols = t.Shape[1]
	}

	result := make([]float64, len(t.Data))

	for r := 0; r < rows; r++ {
		offset := r * cols
		// Find max for numerical stability
		maxVal := t.Data[offset]
		for c := 1; c < cols; c++ {
			if t.Data[offset+c] > maxVal {
				maxVal = t.Data[offset+c]
			}
		}

		// Compute exp and sum
		expSum := 0.0
		for c := 0; c < cols; c++ {
			ex := math.Exp(t.Data[offset+c] - maxVal)
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
				t.Grad[offset+c] += y * (grad - dot)
			}
		}
	}
	return out
}
