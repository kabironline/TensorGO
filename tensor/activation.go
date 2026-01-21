package tensor

import (
	"github.com/kabironline/nanograd/internal/pools"
)

func (t *Tensor) ReLU() *Tensor {
	tContig := Contiguous(t)
	tContigSize := len(tContig.Data)
	result := t.Device.Allocate(tContigSize)
	t.Device.ReLU(tContig.Data, result, tContigSize)

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := t.Device.Allocate(tContigSize)
		t.Device.ReLUBackward(out.Grad, tContig.Data, grad, tContigSize)
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Sigmoid() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Sigmoid(tContig.Data, len(tContig.Data))

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := t.Device.SigmoidBackward(out.Grad, out.Data, len(out.Grad))
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Tanh() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Tanh(tContig.Data, len(tContig.Data))

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		grad := t.Device.TanhBackward(out.Grad, out.Data, len(out.Grad))
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Exp() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Exp(tContig.Data, len(tContig.Data))

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		// dL/dt = dL/dout * exp(t) = dL/dout * out
		grad := t.Device.BroadcastMul(out.Grad, out.Data, out.Shape, out.Shape, out.Shape)
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Log() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Log(tContig.Data, len(tContig.Data))

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		// dL/dt = dL/dout * (1/t)
		grad := t.Device.BroadcastDiv(out.Grad, tContig.Data, out.Shape, t.Shape, out.Shape)
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Square() *Tensor {
	tContig := Contiguous(t)
	result := t.Device.Square(tContig.Data, len(tContig.Data))

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		// dL/dt = dL/dout * 2t
		grad2t := t.Device.MulScalar(tContig.Data, 2.0, len(tContig.Data))
		grad := t.Device.BroadcastMul(out.Grad, grad2t, out.Shape, t.Shape, out.Shape)
		t.AccumulateGrad(grad)
	}
	return out
}

func (t *Tensor) Softmax() *Tensor {
	if len(t.Shape) > 2 {
		panic("Softmax only supports 1D or 2D tensors")
	}

	tContig := Contiguous(t)
	result := t.Device.Softmax(tContig.Data, t.Shape)

	out := NewTensor(result, append([]int{}, t.Shape...), t)
	out.Backward = func() {
		rows := 1
		cols := len(tContig.Data)
		if len(t.Shape) == 2 {
			rows = t.Shape[0]
			cols = t.Shape[1]
		}

		gradInput := pools.GetZeroedBuffer(len(out.Grad))
		for r := 0; r < rows; r++ {
			offset := r * cols
			// Compute sum(y_k * grad_k) for this row
			var dot float32 = 0.0
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
		pools.PutBuffer(gradInput)
	}
	return out
}
