package cpu

import "math"

// PoolOps implementation for CPU

func (b *CPUBackend) MaxPool2d(data []float32, shape, strides []int, kH, kW, stride, padding int) ([]float32, []int) {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]

	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	outSize := N * C * outH * outW
	outData := b.Allocate(outSize)
	maxIndices := make([]int, outSize)

	// Parallelize over N * C
	b.pool.Process(N*C, func(start, end int) {
		for i := start; i < end; i++ {
			n := i / C
			c := i % C
			baseInput := n*strides[0] + c*strides[1]

			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					var maxVal float32 = -math.MaxFloat32
					maxIdx := -1

					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							ih := oh*stride + kh - padding
							iw := ow*stride + kw - padding

							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								idx := baseInput + ih*strides[2] + iw*strides[3]
								val := data[idx]
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
	})

	return outData, maxIndices
}

func (b *CPUBackend) MaxPool2dBackward(grad []float32, indices []int, xShape []int) []float32 {
	xGrad := b.Allocate(xShape[0] * xShape[1] * xShape[2] * xShape[3])

	// Since maxIndices are unique points in the input, we don't need atomic additions
	// if we loop over the grad (output size).
	// However, multiple output pixels could technically point to the same input if
	// MaxPool regions overlap (e.g. stride < kernelSize). In that case, we MUST
	// use atomics or be careful.
	// In PyTorch, MaxPool2d with stride < kernelSize DOES accumulate gradients.

	for i, g := range grad {
		if indices[i] != -1 {
			xGrad[indices[i]] += g
		}
	}

	return xGrad
}
