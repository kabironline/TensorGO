package cpu

import "math"

// Sigmoid applies the sigmoid activation function element-wise: out = 1 / (1 + exp(-x))
func (b *CPUBackend) Sigmoid(a []float64, size int) []float64 {
	out := b.Allocate(size)
	b.pool.Process(size, func(start, end int) {
		for i := start; i < end; i++ {
			out[i] = 1.0 / (1.0 + math.Exp(-a[i]))
		}
	})
	return out
}

// SigmoidBackward computes the gradient of the sigmoid activation: out = grad * sigmoid * (1 - sigmoid)
func (b *CPUBackend) SigmoidBackward(grad, output []float64, size int) []float64 {
	out := b.Allocate(size)
	b.pool.Process(size, func(start, end int) {
		for i := start; i < end; i++ {
			out[i] = grad[i] * output[i] * (1.0 - output[i])
		}
	})
	return out
}
