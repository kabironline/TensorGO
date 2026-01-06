package cpu

import "math"

// Tanh applies the tanh activation function element-wise: out = tanh(x)
func (b *CPUBackend) Tanh(a []float64, size int) []float64 {
	out := b.Allocate(size)
	b.pool.Process(size, func(start, end int) {
		for i := start; i < end; i++ {
			out[i] = math.Tanh(a[i])
		}
	})
	return out
}

// TanhBackward computes the gradient of the tanh activation: out = grad * (1 - tanh^2)
func (b *CPUBackend) TanhBackward(grad, output []float64, size int) []float64 {
	out := b.Allocate(size)
	b.pool.Process(size, func(start, end int) {
		for i := start; i < end; i++ {
			tanhVal := output[i]
			out[i] = grad[i] * (1.0 - tanhVal*tanhVal)
		}
	})
	return out
}
