package activations

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type Tanh struct{}

func (t *Tanh) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Tanh() }
func (t *Tanh) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }

func (t *Tanh) Save(idx int, out map[string]safetensors.TensorView) error {
	data := binary.LittleEndian.AppendUint32(nil, math.Float32bits(0))
	tv, err := safetensors.NewTensorView(safetensors.F32, []uint64{1}, data)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.activation.tanh", idx)] = tv
	return nil
}
