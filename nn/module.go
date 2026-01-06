package nn

import (
	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"

	_ "github.com/kabironline/nanograd/internal/backend/cpu" // Register CPU backend
)

type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
	To(dev backend.Backend)
	// Save writes any parameters or small markers for this module into the provided map.
	// The implementer should use the provided layer index to build stable keys.
	Save(layerIdx int, out map[string]safetensors.TensorView) error
}
