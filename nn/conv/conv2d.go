package conv

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type Conv2D struct {
	Kernel  *tensor.Tensor
	Bias    *tensor.Tensor
	Stride  int
	Padding int

	lastInput  []int          // Shape of the last input tensor
	lastIm2Col *tensor.Tensor // im2col representation of the last input
}

func NewConv2D(inChannels, outChannels, kernelHeight, kernelWidth int) *Conv2D {
	kernelSize := outChannels * inChannels * kernelHeight * kernelWidth
	kernel := tensor.NewTensor(make([]float64, kernelSize), []int{outChannels, inChannels, kernelHeight, kernelWidth})
	bias := tensor.NewTensor(make([]float64, outChannels), []int{outChannels})

	kernel.RandomInit()
	bias.ZeroInit()

	return &Conv2D{
		Kernel:  kernel,
		Bias:    bias,
		Stride:  1,
		Padding: -1,
	}
}

// Forward runs conv on input of shape [N, C, H, W] and returns [N, outC, outH, outW]
func (c *Conv2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Validating input shape
	if len(x.Shape) != 4 {
		panic("Conv2D Forward: input tensor must have 4 dimensions [N, C, H, W]")
	}

	// Compute outH, outW using H, W, kH, kW, stride, padding
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outC := c.Kernel.Shape[0]
	kH, kW := c.Kernel.Shape[2], c.Kernel.Shape[3]

	padding := c.Padding
	if padding < 0 {
		padding = (kH - 1) / 2 // same padding
	}
	stride := c.Stride

	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	// im2col => imCol [C*kH*kW, N*outH*outW]
	imCol := im2col(x, kH, kW, stride, padding)
	imCol.Parents = []*tensor.Tensor{x}
	imCol.Backward = func() {
		x.AccumulateGrad(col2im(imCol.Grad, x.Shape, kH, kW, stride, padding))
	}

	// Kernel is [outC, C, kH, kW]
	// Reshape kernel to [outC, C*kH*kW]
	weightReshaped := c.Kernel.Reshape([]int{outC, C * kH * kW})

	// MatMul: [outC, C*kH*kW] x [C*kH*kW, N*outH*outW] -> [outC, N*outH*outW]
	out := weightReshaped.MatMul(imCol)

	// Reshape to [outC, N, outH, outW]
	out = out.Reshape([]int{outC, N, outH, outW})

	// Transpose to [N, outC, outH, outW]
	out = out.Transpose([]int{1, 0, 2, 3})

	// Add bias
	// Bias is [outC]. We need to broadcast it to [N, outC, outH, outW]
	biasReshaped := c.Bias.Reshape([]int{1, outC, 1, 1})
	out = out.Add(biasReshaped)

	// Setting up backward for out
	out.Parents = []*tensor.Tensor{imCol, c.Kernel, c.Bias}
	c.lastInput = append([]int(nil), x.Shape...)
	c.lastIm2Col = imCol

	out.Backward = func() {
		// nothing to do if no grad
		if out.Grad == nil {
			return
		}

		// bias grad: sum over N, H, W for each out channel
		gradB := make([]float64, outC)
		for n := range N {
			for oc := range outC {
				base := n*(outC*outH*outW) + oc*(outH*outW)
				for h := range outH {
					for w := range outW {
						gradB[oc] += out.Grad[base+h*outW+w]
					}
				}
			}
		}
		c.Bias.AccumulateGrad(gradB)

		// prepare grad matrix gPre with shape [outC, N*outH*outW]
		gT := tensor.NewTensor(out.Grad, []int{N, outC, outH, outW})
		gPre := gT.Transpose([]int{1, 0, 2, 3}).Reshape([]int{outC, N * outH * outW})

		// weight grad = gPre * imCol^T -> shape [outC, C*kH*kW]
		imColT := imCol.Transpose([]int{1, 0})
		gradW := gPre.MatMul(imColT).Reshape(c.Kernel.Shape)
		c.Kernel.AccumulateGrad(gradW.Data)

		// imCol grad = weight^T * gPre -> shape [C*kH*kW, N*outH*outW]
		weightReshaped := c.Kernel.Reshape([]int{outC, C * kH * kW})
		weightT := weightReshaped.Transpose([]int{1, 0})
		gradImCol := weightT.MatMul(gPre)
		imCol.AccumulateGrad(gradImCol.Data)
	}

	return out
}

func im2col(x *tensor.Tensor, kH, kW, stride, padding int) *tensor.Tensor {
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	resData := make([]float64, C*kH*kW*N*outH*outW)
	xData := x.Data
	xStrides := x.Strides
	xOffset := x.Offset

	rowSize := N * outH * outW

	for c := range C {
		cOffset := xOffset + c*xStrides[1]
		for kh := range kH {
			for kw := range kW {
				rowIdx := c*kH*kW + kh*kW + kw
				rowOffset := rowIdx * rowSize
				for n := range N {
					nOffset := cOffset + n*xStrides[0]
					for oh := range outH {
						ih := oh*stride + kh - padding
						if ih >= 0 && ih < H {
							ihOffset := nOffset + ih*xStrides[2]
							for ow := range outW {
								iw := ow*stride + kw - padding
								if iw >= 0 && iw < W {
									colIdx := n*outH*outW + oh*outW + ow
									resData[rowOffset+colIdx] = xData[ihOffset+iw*xStrides[3]]
								}
							}
						}
					}
				}
			}
		}
	}

	return tensor.NewTensor(resData, []int{C * kH * kW, N * outH * outW})
}

func col2im(colGrad []float64, xShape []int, kH, kW, stride, padding int) []float64 {
	N, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	xGrad := make([]float64, N*C*H*W)
	rowSize := N * outH * outW

	for c := range C {
		for kh := range kH {
			for kw := range kW {
				rowIdx := c*kH*kW + kh*kW + kw
				rowOffset := rowIdx * rowSize
				for n := range N {
					for oh := range outH {
						ih := oh*stride + kh - padding
						if ih >= 0 && ih < H {
							for ow := range outW {
								iw := ow*stride + kw - padding
								if iw >= 0 && iw < W {
									colIdx := n*outH*outW + oh*outW + ow
									xGrad[n*C*H*W+c*H*W+ih*W+iw] += colGrad[rowOffset+colIdx]
								}
							}
						}
					}
				}
			}
		}
	}
	return xGrad
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
