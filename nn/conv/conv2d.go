package conv

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type Conv2D struct {
	Kernel  *tensor.Tensor
	Bias    *tensor.Tensor
	Stride  int
	Padding int
}

func NewConv2D(inChannels, outChannels, kernelHeight, kernelWidth int) *Conv2D {
	kernelSize := outChannels * inChannels * kernelHeight * kernelWidth
	kernel := tensor.NewTensor(make([]float32, kernelSize), []int{outChannels, inChannels, kernelHeight, kernelWidth})
	bias := tensor.NewTensor(make([]float32, outChannels), []int{outChannels})

	kernel.RequiresGrad = true
	bias.RequiresGrad = true

	kernel.RandomInit()
	bias.ZeroInit()

	return &Conv2D{
		Kernel:  kernel,
		Bias:    bias,
		Stride:  1,
		Padding: -1,
	}
}

func (c *Conv2D) To(dev backend.Backend) {
	c.Kernel = c.Kernel.To(dev)
	c.Bias = c.Bias.To(dev)
}

// Forward runs conv on input of shape [N, C, H, W] and returns [N, outC, outH, outW]
func (c *Conv2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Validating input shape
	if len(x.Shape) != 4 {
		panic("Conv2D Forward: input tensor must have 4 dimensions [N, C, H, W]")
	}

	// Compute parameters
	outC := c.Kernel.Shape[0]
	kH, kW := c.Kernel.Shape[2], c.Kernel.Shape[3]

	padding := c.Padding
	if padding < 0 {
		padding = (kH - 1) / 2 // same padding
	}
	stride := c.Stride

	// Use backend-accelerated im2col
	imCol := x.Im2Col(kH, kW, stride, padding)

	// Kernel is [outC, C, kH, kW] -> Reshape to [outC, C*kH*kW]
	weightReshaped := c.Kernel.Reshape([]int{outC, -1})

	// MatMul: [outC, C*kH*kW] x [C*kH*kW, N*outH*outW] -> [outC, N*outH*outW]
	out := weightReshaped.MatMul(imCol)

	// Reshape and transpose to final shape [N, outC, outH, outW]
	N := x.Shape[0]
	H, W := x.Shape[2], x.Shape[3]
	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	out = out.Reshape([]int{outC, N, outH, outW})
	out = out.Transpose([]int{1, 0, 2, 3})

	// Add bias if present
	if c.Bias != nil {
		biasReshaped := c.Bias.Reshape([]int{1, outC, 1, 1})
		out = out.Add(biasReshaped)
	}

	return out
}

func (c *Conv2D) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{c.Kernel, c.Bias}
}

func (c *Conv2D) Save(layerIdx int, out map[string]safetensors.TensorView) error {
	// Kernel
	k := c.Kernel
	dataK := make([]byte, 0, tensor.TotalSize(k.Shape)*4)
	for _, v := range k.Data {
		dataK = binary.LittleEndian.AppendUint32(dataK, math.Float32bits(float32(v)))
	}
	shapeK := make([]uint64, len(k.Shape))
	for i, s := range k.Shape {
		shapeK[i] = uint64(s)
	}
	tvK, err := safetensors.NewTensorView(safetensors.F32, shapeK, dataK)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.conv.weight", layerIdx)] = tvK

	// Bias
	b := c.Bias
	dataB := make([]byte, 0, tensor.TotalSize(b.Shape)*4)
	for _, v := range b.Data {
		dataB = binary.LittleEndian.AppendUint32(dataB, math.Float32bits(float32(v)))
	}
	shapeB := make([]uint64, len(b.Shape))
	for i, s := range b.Shape {
		shapeB[i] = uint64(s)
	}
	tvB, err := safetensors.NewTensorView(safetensors.F32, shapeB, dataB)
	if err != nil {
		return err
	}
	out[fmt.Sprintf("layer_%d.conv.bias", layerIdx)] = tvB
	return nil
}
