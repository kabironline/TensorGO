package cpu

import "math"

// Sigmoid applies the sigmoid activation function element-wise: out = 1 / (1 + exp(-x))
func (b *CPUBackend) Sigmoid(a []float32, size int) []float32 {
	out := b.Allocate(size)
	b.pool.Process(size, func(start, end int) {
		for i := start; i < end; i++ {
			out[i] = 1.0 / (1.0 + float32(math.Exp(float64(-a[i]))))
		}
	})
	return out
}

// SigmoidBackward computes the gradient of the sigmoid activation: out = grad * sigmoid * (1 - sigmoid)
func (b *CPUBackend) SigmoidBackward(grad, output []float32, size int) []float32 {
	out := b.Allocate(size)
	b.pool.Process(size, func(start, end int) {
		for i := start; i < end; i++ {
			out[i] = grad[i] * output[i] * (1.0 - output[i])
		}
	})
	return out
}
