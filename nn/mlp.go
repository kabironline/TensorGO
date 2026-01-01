package nn

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

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

// Save saves the model parameters as a safetensor.
func (m MLP) Save(filepath string) error {
	metadata := make(map[string]safetensors.TensorView)

	for i, layer := range m.Layers {
		if err := layer.Save(i, metadata); err != nil {
			return err
		}
	}

	serialized, err := safetensors.Serialize(metadata, nil)
	if err != nil {
		return err
	}

	return os.WriteFile(filepath, serialized, 0644)
}

// Load, function loads the model parameters from a safetensor file and returns a MLP model.
func Load(filepath string) (m MLP, err error) {
	serialized, err := os.ReadFile(filepath)
	if err != nil {
		return MLP{}, err
	}

	loaded, err := safetensors.Deserialize(serialized)
	if err != nil {
		return MLP{}, err
	}

	// Determine max index present
	maxIdx := -1
	for _, name := range loaded.Names() {
		var idx int
		if _, err := fmt.Sscanf(name, "layer_%d.", &idx); err == nil {
			if idx > maxIdx {
				maxIdx = idx
			}
		}
	}

	layers := make([]Module, 0)
	for i := 0; i <= maxIdx; i++ {
		mod, err := LoadModuleAt(i, &loaded)
		if err != nil {
			return MLP{}, err
		}
		if mod != nil {
			layers = append(layers, mod)
		}
	}

	return MLP{Layers: layers}, nil
}

// LoadModuleAt inspects the serialized tensors for a given layer index and
// constructs the appropriate Module (Linear, activation, etc). If no module
// is present at the index it returns (nil, nil).
func LoadModuleAt(idx int, loaded *safetensors.SafeTensors) (Module, error) {
	// Check for linear
	wName := fmt.Sprintf("layer_%d.linear.weight", idx)
	bName := fmt.Sprintf("layer_%d.linear.bias", idx)
	if tvW, ok := loaded.Tensor(wName); ok {
		tvB, okB := loaded.Tensor(bName)
		if !okB {
			return nil, fmt.Errorf("missing bias for linear layer %d", idx)
		}

		// Convert weight
		dataW := tvW.Data()
		floatW := make([]float64, 0, len(dataW)/4)
		for i := 0; i < len(dataW); i += 4 {
			bits := binary.LittleEndian.Uint32(dataW[i : i+4])
			floatW = append(floatW, float64(math.Float32frombits(bits)))
		}
		shapeW := make([]int, len(tvW.Shape()))
		for i, s := range tvW.Shape() {
			shapeW[i] = int(s)
		}

		// Convert bias
		dataB := tvB.Data()
		floatB := make([]float64, 0, len(dataB)/4)
		for i := 0; i < len(dataB); i += 4 {
			bits := binary.LittleEndian.Uint32(dataB[i : i+4])
			floatB = append(floatB, float64(math.Float32frombits(bits)))
		}
		shapeB := make([]int, len(tvB.Shape()))
		for i, s := range tvB.Shape() {
			shapeB[i] = int(s)
		}

		linear := &Linear{
			Weight: tensor.NewTensor(floatW, shapeW),
			Bias:   tensor.NewTensor(floatB, shapeB),
		}
		return linear, nil
	}

	// Check for activations
	for _, act := range activations {
		key := fmt.Sprintf("layer_%d.activation.%s", idx, act)
		if _, ok := loaded.Tensor(key); ok {
			switch act {
			case "relu":
				return &ReLU{}, nil
			case "sigmoid":
				return &Sigmoid{}, nil
			case "tanh":
				return &Tanh{}, nil
			case "softmax":
				return &Softmax{}, nil
			}
		}
	}

	// Nothing found for this index
	return nil, nil
}
