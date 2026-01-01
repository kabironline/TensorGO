package nn

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type Linear struct {
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	w := tensor.NewTensor(make([]float64, inFeatures*outFeatures), []int{inFeatures, outFeatures})
	b := tensor.NewTensor(make([]float64, outFeatures), []int{outFeatures})

	// Initialize weights and biases
	w.RandomInit()
	b.ZeroInit()

	return &Linear{
		Weight: w,
		Bias:   b,
	}
}

func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.MatMul(l.Weight).Add(l.Bias)
}

func (l *Linear) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{l.Weight, l.Bias}
}

// Save writes weight and bias tensors into the provided metadata map using
// keys prefixed with the layer index (e.g. layer_{i}.linear.weight).
func (l *Linear) Save(layerIdx int, out map[string]safetensors.TensorView) error {
	// Weight
	w := l.Weight
	dataW := make([]byte, 0, tensor.TotalSize(w.Shape)*4)
	for _, v := range w.Data {
		dataW = binary.LittleEndian.AppendUint32(dataW, math.Float32bits(float32(v)))
	}
	shapeW := make([]uint64, len(w.Shape))
	for k, s := range w.Shape {
		shapeW[k] = uint64(s)
	}
	tvW, err := safetensors.NewTensorView(safetensors.F32, shapeW, dataW)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.linear.weight", layerIdx)] = tvW

	// Bias
	b := l.Bias
	dataB := make([]byte, 0, tensor.TotalSize(b.Shape)*4)
	for _, v := range b.Data {
		dataB = binary.LittleEndian.AppendUint32(dataB, math.Float32bits(float32(v)))
	}
	shapeB := make([]uint64, len(b.Shape))
	for k, s := range b.Shape {
		shapeB[k] = uint64(s)
	}
	tvB, err := safetensors.NewTensorView(safetensors.F32, shapeB, dataB)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.linear.bias", layerIdx)] = tvB
	return nil
}
