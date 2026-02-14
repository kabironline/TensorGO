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

	xContig := tensor.Contiguous(x)
	wContig := tensor.Contiguous(c.Kernel)

	var bContig *tensor.Tensor
	if c.Bias != nil {
		bContig = tensor.Contiguous(c.Bias)
	}

	var biasData []float32
	if bContig != nil {
		biasData = bContig.Data
	}

	outData := x.Device.Conv2DForward(
		xContig.Data,
		wContig.Data,
		biasData,
		x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3],
		outC, kH, kW,
		padding, padding,
		stride, stride,
	)

	outH := (x.Shape[2]+2*padding-kH)/stride + 1
	outW := (x.Shape[3]+2*padding-kW)/stride + 1
	outShape := []int{x.Shape[0], outC, outH, outW}

	parents := []*tensor.Tensor{x, c.Kernel}
	if c.Bias != nil {
		parents = append(parents, c.Bias)
	}

	out := &tensor.Tensor{
		Data:         outData,
		Shape:        outShape,
		Strides:      tensor.ComputeStrides(outShape),
		Device:       x.Device,
		RequiresGrad: x.RequiresGrad || c.Kernel.RequiresGrad || (c.Bias != nil && c.Bias.RequiresGrad),
		Parents:      parents,
	}

	out.Backward = func() {
		if out.Grad == nil {
			return
		}
		inputGrad, weightGrad, biasGrad := x.Device.Conv2DBackward(
			xContig.Data,
			wContig.Data,
			out.Grad,
			x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3],
			outC, kH, kW,
			padding, padding,
			stride, stride,
		)

		if x.RequiresGrad && inputGrad != nil {
			x.AccumulateGrad(inputGrad)
		}
		if c.Kernel.RequiresGrad && weightGrad != nil {
			c.Kernel.AccumulateGrad(weightGrad)
		}
		if c.Bias != nil && c.Bias.RequiresGrad && biasGrad != nil {
			c.Bias.AccumulateGrad(biasGrad)
		}
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
