package cpu

// Relu applies the ReLU activation function element-wise: out = max(0, x)
func (b *CPUBackend) ReLU(a, out []float32, size int) {
	b.pool.Process(size, func(start, end int) {
		for i := start; i < end; i++ {
			if a[i] > 0 {
				out[i] = a[i]
			} else {
				out[i] = 0
			}
		}
	})
}

// ReLUBackward computes the gradient of the ReLU activation function: out = grad * (input > 0)
func (b *CPUBackend) ReLUBackward(grad, input, out []float32, size int) {
	b.pool.Process(size, func(start, end int) {
		for i := start; i < end; i++ {
			if input[i] > 0 {
				out[i] = grad[i]
			} else {
				out[i] = 0
			}
		}
	})
}
