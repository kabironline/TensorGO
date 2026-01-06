package conv

import (
	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type Flatten struct{}

func NewFlatten() *Flatten {
	return &Flatten{}
}

func (f *Flatten) To(dev backend.Backend) {
	// Stateless
}

func (f *Flatten) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) < 2 {
		return x
	}
	// Keeps the Batch dimension, flattens everything else
	batchSize := x.Shape[0]
	remainingSize := 1
	for _, dim := range x.Shape[1:] {
		remainingSize *= dim
	}
	newShape := []int{batchSize, remainingSize}
	return x.Reshape(newShape)
}

func (f *Flatten) Parameters() []*tensor.Tensor {
	return nil
}

func (f *Flatten) Save(layerIdx int, out map[string]safetensors.TensorView) error {
	// Flatten has no parameters to save
	return nil
}
