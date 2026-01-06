package tensor

// AddScalar returns a new Tensor with each element increased by the scalar value.
func (t *Tensor) AddScalar(scalar float64) *Tensor {
	tContig := Contiguous(t)
	outData := t.Device.AddScalar(tContig.Data, scalar, len(tContig.Data))

	out := NewTensor(outData, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		t.AccumulateGrad(out.Grad)
	}
	return out
}

// SubScalar returns a new Tensor with the scalar value subtracted from each element.
func (t *Tensor) SubScalar(scalar float64) *Tensor {
	tContig := Contiguous(t)
	outData := t.Device.SubScalar(tContig.Data, scalar, len(tContig.Data))

	out := NewTensor(outData, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		t.AccumulateGrad(out.Grad)
	}
	return out
}

// MulScalar returns a new Tensor with each element multiplied by the scalar value.
func (t *Tensor) MulScalar(scalar float64) *Tensor {
	tContig := Contiguous(t)
	outData := t.Device.MulScalar(tContig.Data, scalar, len(tContig.Data))

	out := NewTensor(outData, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := t.Device.MulScalar(out.Grad, scalar, len(out.Grad))
		t.AccumulateGrad(grad)
	}
	return out
}

// DivScalar returns a new Tensor with each element divided by the scalar value.
func (t *Tensor) DivScalar(scalar float64) *Tensor {
	tContig := Contiguous(t)
	outData := t.Device.DivScalar(tContig.Data, scalar, len(tContig.Data))

	out := NewTensor(outData, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := t.Device.DivScalar(out.Grad, scalar, len(out.Grad))
		t.AccumulateGrad(grad)
	}
	return out
}
