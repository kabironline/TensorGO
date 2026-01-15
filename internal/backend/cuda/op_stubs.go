package cuda

// ============================================================================
// Element-wise Operations (Minimal set to satisfy interface)
// ============================================================================

func (b *CUDABackend) Add(a, b_val []float64, size int) []float64                { return nil }
func (b *CUDABackend) Sub(a, b_val []float64, size int) []float64                { return nil }
func (b *CUDABackend) Mul(a, b_val []float64, size int) []float64                { return nil }
func (b *CUDABackend) Div(a, b_val []float64, size int) []float64                { return nil }
func (b *CUDABackend) Neg(a []float64, size int) []float64                       { return nil }
func (b *CUDABackend) AddScalar(a []float64, scalar float64, size int) []float64 { return nil }
func (b *CUDABackend) SubScalar(a []float64, scalar float64, size int) []float64 { return nil }
func (b *CUDABackend) MulScalar(a []float64, scalar float64, size int) []float64 { return nil }
func (b *CUDABackend) DivScalar(a []float64, scalar float64, size int) []float64 { return nil }

func (b *CUDABackend) Pow(a []float64, power float64, size int) []float64 { return nil }
func (b *CUDABackend) Exp(a []float64, size int) []float64                { return nil }
func (b *CUDABackend) Log(a []float64, size int) []float64                { return nil }
func (b *CUDABackend) Sqrt(a []float64, size int) []float64               { return nil }
func (b *CUDABackend) Square(a []float64, size int) []float64             { return nil }

// Matrix Operations
// func (b *CUDABackend) MatMul(a, b_val []float64, m, n, k, strideA, strideB int) []float64 { return nil }
func (b *CUDABackend) MatMulAdd(a, b_val, c []float64, m, n, k, strideA, strideB int) []float64 {
	return nil
}
func (b *CUDABackend) MatMulTransA(a, b_val, out []float64, m, n, k, strideA, strideB int) []float64 {
	return nil
}
func (b *CUDABackend) MatMulTransB(a, b_val, out []float64, m, n, k, strideA, strideB int) []float64 {
	return nil
}

// Reduction Operations
func (b *CUDABackend) Sum(data []float64, size int) float64  { return 0 }
func (b *CUDABackend) Mean(data []float64, size int) float64 { return 0 }
func (b *CUDABackend) Max(data []float64, size int) float64  { return 0 }
func (b *CUDABackend) Min(data []float64, size int) float64  { return 0 }

// Axis Reductions
func (b *CUDABackend) SumAxis(data []float64, shape []int, axis int) []float64  { return nil }
func (b *CUDABackend) MeanAxis(data []float64, shape []int, axis int) []float64 { return nil }
func (b *CUDABackend) MaxAxis(data []float64, shape []int, axis int) []float64  { return nil }

// NN Ops
func (b *CUDABackend) ReLU(a []float64, size int) []float64                   { return nil }
func (b *CUDABackend) ReLUBackward(grad, input []float64, size int) []float64 { return nil }
func (b *CUDABackend) Sigmoid(a []float64, size int) []float64                { return nil }
func (b *CUDABackend) SigmoidBackward(grad, output []float64, size int) []float64 {
	return nil
}
func (b *CUDABackend) Tanh(a []float64, size int) []float64 { return nil }
func (b *CUDABackend) TanhBackward(grad, output []float64, size int) []float64 {
	return nil
}
func (b *CUDABackend) Softmax(data []float64, shape []int) []float64 { return nil }
func (b *CUDABackend) SoftmaxBackward(grad, output []float64, shape []int) []float64 {
	return nil
}
func (b *CUDABackend) LogSoftmax(data []float64, shape []int) []float64 { return nil }

// Broadcast
func (b *CUDABackend) BroadcastAdd(a, b_val []float64, aShape, bShape, outShape []int) []float64 {
	return nil
}
func (b *CUDABackend) BroadcastSub(a, b_val []float64, aShape, bShape, outShape []int) []float64 {
	return nil
}
func (b *CUDABackend) BroadcastMul(a, b_val []float64, aShape, bShape, outShape []int) []float64 {
	return nil
}
func (b *CUDABackend) BroadcastDiv(a, b_val []float64, aShape, bShape, outShape []int) []float64 {
	return nil
}

// Utility
func (b *CUDABackend) Fill(data []float64, value float64, size int) {}
func (b *CUDABackend) Clone(data []float64, size int) []float64     { return nil }
func (b *CUDABackend) Transpose(a []float64, rows, cols int) []float64 {
	return nil
}
func (b *CUDABackend) Contiguous(data []float64, shape, strides []int) []float64 { return nil }

// Conv
func (b *CUDABackend) Im2Col(data []float64, shape, strides []int, kH, kW, stride, padding int) []float64 {
	return nil
}
func (b *CUDABackend) Col2Im(colGrad []float64, xShape, xStrides []int, kH, kW, stride, padding int) []float64 {
	return nil
}

// Pool
func (b *CUDABackend) MaxPool2d(data []float64, shape, strides []int, kH, kW, stride, padding int) ([]float64, []int) {
	return nil, nil
}
func (b *CUDABackend) MaxPool2dBackward(grad []float64, indices []int, xShape []int) []float64 {
	return nil
}

// Random
func (b *CUDABackend) Normal(data []float64, mean, stdDev float64, size int) {}

// Optimizer
func (b *CUDABackend) StepSGD(data, grad []float64, lr float64)                                  {}
func (b *CUDABackend) StepAdam(data, grad, m, v []float64, lr, beta1, beta2, eps float64, t int) {}
