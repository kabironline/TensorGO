package nn

import (
	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
	// Save writes any parameters or small markers for this module into the provided map.
	// The implementer should use the provided layer index to build stable keys.
	Save(layerIdx int, out map[string]safetensors.TensorView) error
}
