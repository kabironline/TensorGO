package conv

import (
	"math"

	"github.com/kabironline/nanograd/tensor"
	"github.com/nlpodyssey/safetensors"
)

type MaxPool2D struct {
	KernelSize int
	Stride     int
	Padding    int
}

func NewMaxPool2D(kernelSize, stride, padding int) *MaxPool2D {
	return &MaxPool2D{
		KernelSize: kernelSize,
		Stride:     stride,
		Padding:    padding,
	}
}

func (m *MaxPool2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("MaxPool2D Forward: input tensor must have 4 dimensions [N, C, H, W]")
	}

	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	kH, kW := m.KernelSize, m.KernelSize
	stride := m.Stride
	padding := m.Padding

	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	outData := make([]float64, N*C*outH*outW)
	maxIndices := make([]int, N*C*outH*outW)

	for n := range N {
		for c := range C {
			for oh := range outH {
				for ow := range outW {
					maxVal := -math.MaxFloat64
					maxIdx := -1

					for kh := range kH {
						for kw := range kW {
							ih := oh*stride + kh - padding
							iw := ow*stride + kw - padding

							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								idx := n*C*H*W + c*H*W + ih*W + iw
								val := x.Data[idx]
								if val > maxVal {
									maxVal = val
									maxIdx = idx
								}
							}
						}
					}
					outIdx := n*C*outH*outW + c*outH*outW + oh*outW + ow
					outData[outIdx] = maxVal
					maxIndices[outIdx] = maxIdx
				}
			}
		}
	}

	out := tensor.NewTensor(outData, []int{N, C, outH, outW}, x)
	out.Backward = func() {
		if out.Grad == nil {
			return
		}
		xGrad := make([]float64, len(x.Data))
		for i, g := range out.Grad {
			if maxIndices[i] != -1 {
				xGrad[maxIndices[i]] += g
			}
		}
		x.AccumulateGrad(xGrad)
	}

	return out
}

func (m *MaxPool2D) Parameters() []*tensor.Tensor {
	return nil
}

func (m *MaxPool2D) Save(layerIdx int, out map[string]safetensors.TensorView) error {
	// MaxPool2D has no parameters to save
	return nil
}
