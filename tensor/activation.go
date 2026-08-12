package tensor

func (t *Tensor) ReLU() *Tensor {
	tContig := Contiguous(t)
	tContigSize := len(tContig.Data())
	result := t.Device.Allocate(tContigSize)
	t.Device.ReLU(tContig.Data(), result, tContigSize)

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(result),
		Shape:        append([]int{}, t.Shape...),
		Strides:      ComputeStrides(t.Shape),
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Parents:      []*Tensor{t},
		contiguous:   true,
	}

	out.Backward = func() {
		grad := t.Device.Allocate(tContigSize)
		t.Device.ReLUBackward(out.Grad(), tContig.Data(), grad, tContigSize)
		t.AccumulateGrad(grad)
		t.Device.Free(grad)
	}
	return out
}

func (t *Tensor) Sigmoid() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Sigmoid(tContig.Data(), len(tContig.Data()))

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(result),
		Shape:        append([]int{}, t.Shape...),
		Strides:      ComputeStrides(t.Shape),
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Parents:      []*Tensor{t},
		contiguous:   true,
	}

	out.Backward = func() {
		grad := t.Device.SigmoidBackward(out.Grad(), out.Data(), len(out.Grad()))
		t.AccumulateGrad(grad)
		t.Device.Free(grad)
	}
	return out
}

func (t *Tensor) Tanh() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Tanh(tContig.Data(), len(tContig.Data()))

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(result),
		Shape:        append([]int{}, t.Shape...),
		Strides:      ComputeStrides(t.Shape),
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Parents:      []*Tensor{t},
		contiguous:   true,
	}

	out.Backward = func() {
		grad := t.Device.TanhBackward(out.Grad(), out.Data(), len(out.Grad()))
		t.AccumulateGrad(grad)
		t.Device.Free(grad)
	}
	return out
}

func (t *Tensor) Exp() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Exp(tContig.Data(), len(tContig.Data()))

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(result),
		Shape:        append([]int{}, t.Shape...),
		Strides:      ComputeStrides(t.Shape),
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Parents:      []*Tensor{t},
		contiguous:   true,
	}

	out.Backward = func() {
		// dL/dt = dL/dout * exp(t) = dL/dout * out
		grad := t.Device.BroadcastMul(out.Grad(), out.Data(), out.Shape, out.Shape, out.Shape)
		t.AccumulateGrad(grad)
		t.Device.Free(grad)
	}
	return out
}

func (t *Tensor) Log() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Log(tContig.Data(), len(tContig.Data()))

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(result),
		Shape:        append([]int{}, t.Shape...),
		Strides:      ComputeStrides(t.Shape),
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Parents:      []*Tensor{t},
		contiguous:   true,
	}

	out.Backward = func() {
		// dL/dt = dL/dout * (1/t)
		grad := t.Device.BroadcastDiv(out.Grad(), tContig.Data(), out.Shape, t.Shape, out.Shape)
		t.AccumulateGrad(grad)
		t.Device.Free(grad)
	}
	return out
}

func (t *Tensor) Square() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Square(tContig.Data(), len(tContig.Data()))

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(result),
		Shape:        append([]int{}, t.Shape...),
		Strides:      ComputeStrides(t.Shape),
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Parents:      []*Tensor{t},
		contiguous:   true,
	}

	out.Backward = func() {
		// dL/dt = dL/dout * 2t
		grad2t := t.Device.MulScalar(tContig.Data(), 2.0, len(tContig.Data()))
		grad := t.Device.BroadcastMul(out.Grad(), grad2t, out.Shape, t.Shape, out.Shape)
		t.AccumulateGrad(grad)
		t.Device.Free(grad)
		t.Device.Free(grad2t)
	}
	return out
}

func (t *Tensor) Softmax() *Tensor {
	if len(t.Shape) > 2 {
		panic("Softmax only supports 1D or 2D tensors")
	}

	tContig := Contiguous(t)
	result := t.Device.Softmax(tContig.Data(), t.Shape)

	// Create output tensor manually to avoid ToDevice being called on GPU memory
	out := &Tensor{
		data:         StorageFrom(result),
		Shape:        append([]int{}, t.Shape...),
		Strides:      ComputeStrides(t.Shape),
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Parents:      []*Tensor{t},
		contiguous:   true,
	}

	out.Backward = func() {
		// Use backend's SoftmaxBackward operation
		grad := t.Device.SoftmaxBackward(out.Grad(), out.Data(), t.Shape)
		t.AccumulateGrad(grad)
		t.Device.Free(grad)
	}
	return out
}
