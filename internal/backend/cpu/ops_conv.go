package cpu

// ConvOps implementation for CPU

func (b *CPUBackend) Im2Col(data []float32, shape, strides []int, kH, kW, stride, padding int) []float32 {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	resSize := C * kH * kW * N * outH * outW
	resData := b.Allocate(resSize)
	rowSize := N * outH * outW

	// Parallelize over channels for better CPU utilization
	b.pool.Process(C, func(startC, endC int) {
		for c := startC; c < endC; c++ {
			cOffset := c * strides[1]
			for kh := 0; kh < kH; kh++ {
				for kw := 0; kw < kW; kw++ {
					rowIdx := c*kH*kW + kh*kW + kw
					rowOffset := rowIdx * rowSize
					for n := 0; n < N; n++ {
						nOffset := cOffset + n*strides[0]
						for oh := 0; oh < outH; oh++ {
							ih := oh*stride + kh - padding
							if ih >= 0 && ih < H {
								ihOffset := nOffset + ih*strides[2]
								for ow := 0; ow < outW; ow++ {
									iw := ow*stride + kw - padding
									if iw >= 0 && iw < W {
										colIdx := n*outH*outW + oh*outW + ow
										resData[rowOffset+colIdx] = data[ihOffset+iw*strides[3]]
									}
								}
							}
						}
					}
				}
			}
		}
	})

	return resData
}

func (b *CPUBackend) Col2Im(colGrad []float32, xShape, xStrides []int, kH, kW, stride, padding int) []float32 {
	N, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	if N*C*H*W == 0 {
		return make([]float32, 0)
	}

	// Double check for zero-size to avoid panic in next loop
	if len(colGrad) == 0 {
		return b.Allocate(N * C * H * W)
	}

	xGrad := b.Allocate(N * C * H * W)
	rowSize := N * outH * outW

	b.pool.Process(C, func(startC, endC int) {
		if startC >= endC {
			return
		}
		for c := startC; c < endC; c++ {
			for kh := 0; kh < kH; kh++ {
				for kw := 0; kw < kW; kw++ {
					rowIdx := c*kH*kW + kh*kW + kw
					rowOffset := rowIdx * rowSize
					for n := 0; n < N; n++ {
						for oh := 0; oh < outH; oh++ {
							ih := oh*stride + kh - padding
							if ih >= 0 && ih < H {
								for ow := 0; ow < outW; ow++ {
									iw := ow*stride + kw - padding
									if iw >= 0 && iw < W {
										colIdx := n*outH*outW + oh*outW + ow
										// Gradient accumulation back to original layout
										// Col2Im always writes to a contiguous xGrad for the next layer
										xGrad[n*C*H*W+c*H*W+ih*W+iw] += colGrad[rowOffset+colIdx]
									}
								}
							}
						}
					}
				}
			}
		}
	})
	return xGrad
}
