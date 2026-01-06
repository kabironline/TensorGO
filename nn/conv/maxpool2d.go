package conv

import (
	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type MaxPool2D struct {
	KernelSize int
	Stride     int
	Padding    int
}

func NewMaxPool2D(kernelSize, stride, padding int) *MaxPool2D {
	return &MaxPool2D{
		KernelSize: kernelSize,
		Stride:     stride,
		Padding:    padding,
	}
}

func (m *MaxPool2D) To(dev backend.Backend) {
	// Stateless
}

func (m *MaxPool2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("MaxPool2D Forward: input tensor must have 4 dimensions [N, C, H, W]")
	}

	return x.MaxPool2D(m.KernelSize, m.KernelSize, m.Stride, m.Padding)
}

func (m *MaxPool2D) Parameters() []*tensor.Tensor {
	return nil
}

func (m *MaxPool2D) Save(layerIdx int, out map[string]safetensors.TensorView) error {
	// MaxPool2D has no parameters to save
	return nil
}
