package cpu

func contiguousStrides4D(n, c, h, w int) []int {
	return []int{c * h * w, h * w, w, 1}
}

func im2colCPU(b *CPUBackend, data []float32, shape, strides []int, kH, kW, stride, padding int) []float32 {
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

func col2imCPU(b *CPUBackend, colGrad []float32, xShape []int, kH, kW, stride, padding int) []float32 {
	N, C, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	outH := (H+2*padding-kH)/stride + 1
	outW := (W+2*padding-kW)/stride + 1

	if N*C*H*W == 0 {
		return make([]float32, 0)
	}

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

func (b *CPUBackend) Conv2DForward(
	input, weights, bias []float32,
	batchSize, inChannels, inHeight, inWidth int,
	outChannels, kernelHeight, kernelWidth int,
	padHeight, padWidth int,
	strideHeight, strideWidth int,
) []float32 {
	if len(input) == 0 || len(weights) == 0 {
		return nil
	}
	if batchSize <= 0 || inChannels <= 0 || inHeight <= 0 || inWidth <= 0 {
		return nil
	}
	if outChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0 {
		return nil
	}

	outH := (inHeight+2*padHeight-kernelHeight)/strideHeight + 1
	outW := (inWidth+2*padWidth-kernelWidth)/strideWidth + 1
	if outH <= 0 || outW <= 0 {
		return nil
	}

	rowSize := batchSize * outH * outW
	k := inChannels * kernelHeight * kernelWidth

	strides := contiguousStrides4D(batchSize, inChannels, inHeight, inWidth)
	cols := im2colCPU(b, input, []int{batchSize, inChannels, inHeight, inWidth}, strides, kernelHeight, kernelWidth, strideHeight, padHeight)

	matOut := b.Allocate(outChannels * rowSize)
	b.MatMul(weights, cols, matOut, outChannels, rowSize, k, k, rowSize)

	out := b.Allocate(batchSize * outChannels * outH * outW)
	useBias := len(bias) == outChannels

	b.pool.Process(outChannels, func(startC, endC int) {
		for oc := startC; oc < endC; oc++ {
			biasVal := float32(0)
			if useBias {
				biasVal = bias[oc]
			}
			baseMat := oc * rowSize
			for n := 0; n < batchSize; n++ {
				baseOut := n*outChannels*outH*outW + oc*outH*outW
				baseCol := n * outH * outW
				for oh := 0; oh < outH; oh++ {
					rowOff := oh * outW
					for ow := 0; ow < outW; ow++ {
						idx := rowOff + ow
						out[baseOut+idx] = matOut[baseMat+baseCol+idx] + biasVal
					}
				}
			}
		}
	})

	return out
}

func (b *CPUBackend) Conv2DBackward(
	input, weights, outputGrad []float32,
	batchSize, inChannels, inHeight, inWidth int,
	outChannels, kernelHeight, kernelWidth int,
	padHeight, padWidth int,
	strideHeight, strideWidth int,
) (inputGrad, weightsGrad, biasGrad []float32) {
	if len(input) == 0 || len(weights) == 0 || len(outputGrad) == 0 {
		return nil, nil, nil
	}
	if batchSize <= 0 || inChannels <= 0 || inHeight <= 0 || inWidth <= 0 {
		return nil, nil, nil
	}
	if outChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0 {
		return nil, nil, nil
	}

	outH := (inHeight+2*padHeight-kernelHeight)/strideHeight + 1
	outW := (inWidth+2*padWidth-kernelWidth)/strideWidth + 1
	if outH <= 0 || outW <= 0 {
		return nil, nil, nil
	}

	rowSize := batchSize * outH * outW
	k := inChannels * kernelHeight * kernelWidth

	strides := contiguousStrides4D(batchSize, inChannels, inHeight, inWidth)
	cols := im2colCPU(b, input, []int{batchSize, inChannels, inHeight, inWidth}, strides, kernelHeight, kernelWidth, strideHeight, padHeight)

	gradMat := b.Allocate(outChannels * rowSize)
	b.pool.Process(outChannels, func(startC, endC int) {
		for oc := startC; oc < endC; oc++ {
			baseMat := oc * rowSize
			for n := 0; n < batchSize; n++ {
				baseOut := n*outChannels*outH*outW + oc*outH*outW
				baseCol := n * outH * outW
				for oh := 0; oh < outH; oh++ {
					rowOff := oh * outW
					for ow := 0; ow < outW; ow++ {
						idx := rowOff + ow
						gradMat[baseMat+baseCol+idx] = outputGrad[baseOut+idx]
					}
				}
			}
		}
	})

	weightsGrad = b.Allocate(outChannels * k)
	b.MatMulTransB(gradMat, cols, weightsGrad, outChannels, k, rowSize, rowSize, rowSize)

	colGrad := b.Allocate(k * rowSize)
	b.MatMulTransA(weights, gradMat, colGrad, k, rowSize, outChannels, k, rowSize)
	inputGrad = col2imCPU(b, colGrad, []int{batchSize, inChannels, inHeight, inWidth}, kernelHeight, kernelWidth, strideHeight, padHeight)

	biasGrad = b.Allocate(outChannels)
	b.pool.Process(outChannels, func(startC, endC int) {
		for oc := startC; oc < endC; oc++ {
			var sum float32
			for n := 0; n < batchSize; n++ {
				baseOut := n*outChannels*outH*outW + oc*outH*outW
				for i := 0; i < outH*outW; i++ {
					sum += outputGrad[baseOut+i]
				}
			}
			biasGrad[oc] = sum
		}
	})

	return inputGrad, weightsGrad, biasGrad
}
