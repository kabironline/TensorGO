package activations

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type ReLU struct{}

func (r *ReLU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.ReLU() }
func (r *ReLU) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }
func (r *ReLU) To(dev backend.Backend)                  {}

func (r *ReLU) Save(idx int, out map[string]safetensors.TensorView) error {
	data := binary.LittleEndian.AppendUint32(nil, math.Float32bits(0))
	tv, err := safetensors.NewTensorView(safetensors.F32, []uint64{1}, data)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.activation.relu", idx)] = tv
	return nil
}
