package tensor

// AddScalar returns a new Tensor with each element increased by the scalar value.
func (t *Tensor) AddScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	// Note: This iterates over physical data. If t is a view, this might process extra data
	// or be in wrong order if we just used result directly.
	// However, NewTensor creates a contiguous tensor.
	// Correct approach: Iterate logically.

	tContig := Contiguous(t)
	for i, v := range tContig.Data {
		result[i] = v + scalar
	}

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		t.AccumulateGrad(out.Grad)
	}
	return out
}

// SubScalar returns a new Tensor with the scalar value subtracted from each element.
func (t *Tensor) SubScalar(scalar float64) *Tensor {
	tContig := Contiguous(t)
	result := make([]float64, len(tContig.Data))
	for i, v := range tContig.Data {
		result[i] = v - scalar
	}
	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		t.AccumulateGrad(out.Grad)
	}
	return out
}

// MulScalar returns a new Tensor with each element multiplied by the scalar value.
func (t *Tensor) MulScalar(scalar float64) *Tensor {
	tContig := Contiguous(t)
	result := make([]float64, len(tContig.Data))
	for i, v := range tContig.Data {
		result[i] = v * scalar
	}
	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := make([]float64, len(out.Grad))
		for i, g := range out.Grad {
			grad[i] = g * scalar
		}
		t.AccumulateGrad(grad)
	}
	return out
}

// DivScalar returns a new Tensor with each element divided by the scalar value.
func (t *Tensor) DivScalar(scalar float64) *Tensor {
	tContig := Contiguous(t)
	result := make([]float64, len(tContig.Data))
	for i, v := range tContig.Data {
		result[i] = v / scalar
	}
	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := make([]float64, len(out.Grad))
		for i, g := range out.Grad {
			grad[i] = g / scalar
		}
		t.AccumulateGrad(grad)
	}
	return out
}
