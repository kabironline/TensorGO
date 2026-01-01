package nn

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

var activations = []string{
	"relu",
	"sigmoid",
	"tanh",
	"softmax",
}

type ReLU struct{}

func (r *ReLU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.ReLU() }
func (r *ReLU) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }

func (r *ReLU) Save(idx int, out map[string]safetensors.TensorView) error {
	data := binary.LittleEndian.AppendUint32(nil, math.Float32bits(0))
	tv, err := safetensors.NewTensorView(safetensors.F32, []uint64{1}, data)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.activation.relu", idx)] = tv
	return nil
}

type Sigmoid struct{}

func (s *Sigmoid) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Sigmoid() }
func (s *Sigmoid) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }

func (s *Sigmoid) Save(idx int, out map[string]safetensors.TensorView) error {
	data := binary.LittleEndian.AppendUint32(nil, math.Float32bits(0))
	tv, err := safetensors.NewTensorView(safetensors.F32, []uint64{1}, data)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.activation.sigmoid", idx)] = tv
	return nil
}

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

type Softmax struct{}

func (t *Softmax) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softmax() }
func (t *Softmax) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }

func (t *Softmax) Save(idx int, out map[string]safetensors.TensorView) error {
	data := binary.LittleEndian.AppendUint32(nil, math.Float32bits(0))
	tv, err := safetensors.NewTensorView(safetensors.F32, []uint64{1}, data)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.activation.softmax", idx)] = tv
	return nil
}
