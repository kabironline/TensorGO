package nn

import "github.com/kabironline/nanograd/tensor"

type MLP struct {
	Layers []Module
}

func NewMLP(inputDim int, outputDims []int, outputActivations []Module) *MLP {
	if len(outputActivations) != len(outputDims) {
		panic("NewMLP: outputActivations length must be equal to outputDims length")
	}
	layers := make([]Module, 0, 2*len(outputDims))
	prevDim := inputDim
	for i := range outputDims {
		layers = append(layers, NewLinear(prevDim, outputDims[i]), outputActivations[i])
		prevDim = outputDims[i]
	}
	return &MLP{
		Layers: layers,
	}
}

func (m MLP) Forward(x *tensor.Tensor) *tensor.Tensor {
	out := x
	for _, layer := range m.Layers {
		out = layer.Forward(out)
	}
	return out
}

func (m MLP) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
