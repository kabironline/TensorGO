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
	maxVal := t.Data[0]
	for _, v := range t.Data {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	expVals := make([]float64, len(t.Data))
	for i, v := range t.Data {
		expVals[i] = math.Exp(v - maxVal) // for numerical stability
		expSum += expVals[i]
	}

	result := make([]float64, len(t.Data))
	for i, v := range expVals {
		result[i] = v / expSum
	}
	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		for i := range out.Data {
			// Simplified gradient for softmax (not full Jacobian)
			t.Grad[i] += out.Grad[i] * out.Data[i] * (1 - out.Data[i])
		}
	}
	return out
}
