package tensor

// MaxPool2D performs 2D max pooling
func (t *Tensor) MaxPool2D(kH, kW, stride, padding int) *Tensor {
	if len(t.Shape) != 4 {
		panic("MaxPool2D requires a 4D tensor [N, C, H, W]")
	}

	resData, indices := t.Device.MaxPool2d(t.Data, t.Shape, t.Strides, kH, kW, stride, padding)

	N, C, H, W := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	out := NewTensor(resData, []int{N, C, outH, outW}, t)

	out.Backward = func() {
		gradT := t.Device.MaxPool2dBackward(out.Grad, indices, t.Shape)
		t.AccumulateGrad(gradT)
	}

	return out
}
