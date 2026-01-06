package activations

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type Softmax struct{}

func (s *Softmax) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softmax() }
func (s *Softmax) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }
func (s *Softmax) To(dev backend.Backend)                  {}

func (t *Softmax) Save(idx int, out map[string]safetensors.TensorView) error {
	data := binary.LittleEndian.AppendUint32(nil, math.Float32bits(0))
	tv, err := safetensors.NewTensorView(safetensors.F32, []uint64{1}, data)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.activation.softmax", idx)] = tv
	return nil
}
