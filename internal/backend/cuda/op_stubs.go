package cuda

// ============================================================================
// Element-wise Operations (Minimal set to satisfy interface)
// ============================================================================

func (b *CUDABackend) Sub(a, b_val []float32, size int) []float32                { return nil }
func (b *CUDABackend) Mul(a, b_val []float32, size int) []float32                { return nil }
func (b *CUDABackend) Div(a, b_val []float32, size int) []float32                { return nil }
func (b *CUDABackend) Neg(a []float32, size int) []float32                       { return nil }
func (b *CUDABackend) AddScalar(a []float32, scalar float32, size int) []float32 { return nil }
func (b *CUDABackend) SubScalar(a []float32, scalar float32, size int) []float32 { return nil }
func (b *CUDABackend) MulScalar(a []float32, scalar float32, size int) []float32 { return nil }
func (b *CUDABackend) DivScalar(a []float32, scalar float32, size int) []float32 { return nil }

func (b *CUDABackend) Pow(a []float32, power float32, size int) []float32 { return nil }
func (b *CUDABackend) Exp(a []float32, size int) []float32                { return nil }
func (b *CUDABackend) Log(a []float32, size int) []float32                { return nil }
func (b *CUDABackend) Sqrt(a []float32, size int) []float32               { return nil }
func (b *CUDABackend) Square(a []float32, size int) []float32             { return nil }

// Matrix Operations
// func (b *CUDABackend) MatMul(a, b_val []float32, m, n, k, strideA, strideB int) []float32 { return nil }
func (b *CUDABackend) MatMulAdd(a, b_val, c []float32, m, n, k, strideA, strideB int) []float32 {
	return nil
}
func (b *CUDABackend) MatMulTransA(a, b_val, out []float32, m, n, k, strideA, strideB int) []float32 {
	return nil
}
func (b *CUDABackend) MatMulTransB(a, b_val, out []float32, m, n, k, strideA, strideB int) []float32 {
	return nil
}

// Reduction Operations
func (b *CUDABackend) Sum(data []float32, size int) float32  { return 0 }
func (b *CUDABackend) Mean(data []float32, size int) float32 { return 0 }
func (b *CUDABackend) Max(data []float32, size int) float32  { return 0 }
func (b *CUDABackend) Min(data []float32, size int) float32  { return 0 }

// Axis Reductions
func (b *CUDABackend) SumAxis(data []float32, shape []int, axis int) []float32  { return nil }
func (b *CUDABackend) MeanAxis(data []float32, shape []int, axis int) []float32 { return nil }
func (b *CUDABackend) MaxAxis(data []float32, shape []int, axis int) []float32  { return nil }

// NN Ops
func (b *CUDABackend) ReLU(a []float32, size int) []float32                   { return nil }
func (b *CUDABackend) ReLUBackward(grad, input []float32, size int) []float32 { return nil }
func (b *CUDABackend) Sigmoid(a []float32, size int) []float32                { return nil }
func (b *CUDABackend) SigmoidBackward(grad, output []float32, size int) []float32 {
	return nil
}
func (b *CUDABackend) Tanh(a []float32, size int) []float32 { return nil }
func (b *CUDABackend) TanhBackward(grad, output []float32, size int) []float32 {
	return nil
}
func (b *CUDABackend) Softmax(data []float32, shape []int) []float32 { return nil }
func (b *CUDABackend) SoftmaxBackward(grad, output []float32, shape []int) []float32 {
	return nil
}
func (b *CUDABackend) LogSoftmax(data []float32, shape []int) []float32 { return nil }

// Broadcast
func (b *CUDABackend) BroadcastAdd(a, b_val []float32, aShape, bShape, outShape []int) []float32 {
	return nil
}
func (b *CUDABackend) BroadcastSub(a, b_val []float32, aShape, bShape, outShape []int) []float32 {
	return nil
}
func (b *CUDABackend) BroadcastMul(a, b_val []float32, aShape, bShape, outShape []int) []float32 {
	return nil
}
func (b *CUDABackend) BroadcastDiv(a, b_val []float32, aShape, bShape, outShape []int) []float32 {
	return nil
}

// Utility
func (b *CUDABackend) Fill(data []float32, value float32, size int) {}
func (b *CUDABackend) Clone(data []float32, size int) []float32     { return nil }
func (b *CUDABackend) Transpose(a []float32, rows, cols int) []float32 {
	return nil
}
func (b *CUDABackend) Contiguous(data []float32, shape, strides []int) []float32 { return nil }

// Conv
func (b *CUDABackend) Im2Col(data []float32, shape, strides []int, kH, kW, stride, padding int) []float32 {
	return nil
}
func (b *CUDABackend) Col2Im(colGrad []float32, xShape, xStrides []int, kH, kW, stride, padding int) []float32 {
	return nil
}

// Pool
func (b *CUDABackend) MaxPool2d(data []float32, shape, strides []int, kH, kW, stride, padding int) ([]float32, []int) {
	return nil, nil
}
func (b *CUDABackend) MaxPool2dBackward(grad []float32, indices []int, xShape []int) []float32 {
	return nil
}

// Random
func (b *CUDABackend) Normal(data []float32, mean, stdDev float32, size int) {}

// Optimizer
func (b *CUDABackend) StepSGD(data, grad []float32, lr float32)                                  {}
func (b *CUDABackend) StepAdam(data, grad, m, v []float32, lr, beta1, beta2, eps float32, t int) {}
