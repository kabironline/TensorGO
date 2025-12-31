package tensor

// AddScalar returns a new Tensor with each element increased by the scalar value.
func (t *Tensor) AddScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v + scalar
	}
	return NewTensor(result, append([]int{}, t.Shape...))
}

// SubScalar returns a new Tensor with the scalar value subtracted from each element.
func (t *Tensor) SubScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v - scalar
	}
	return NewTensor(result, append([]int{}, t.Shape...))
}

// MulScalar returns a new Tensor with each element multiplied by the scalar value.
func (t *Tensor) MulScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v * scalar
	}
	return NewTensor(result, append([]int{}, t.Shape...))
}

// DivScalar returns a new Tensor with each element divided by the scalar value.
func (t *Tensor) DivScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v / scalar
	}
	return NewTensor(result, append([]int{}, t.Shape...))
}
