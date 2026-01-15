package cpu

import (
	"math"
)

// StepSGD performs a single step of SGD update: data -= lr * grad
func (b *CPUBackend) StepSGD(data, grad []float32, lr float32) {
	// Simple element-wise update. Could be parallelized for large tensors.
	if len(data) > b.parallelThreshold {
		b.pool.Process(len(data), func(start, end int) {
			for i := start; i < end; i++ {
				data[i] -= lr * grad[i]
			}
		})
	} else {
		for i := range data {
			data[i] -= lr * grad[i]
		}
	}
}

// StepAdam performs a single step of Adam update
func (b *CPUBackend) StepAdam(data, grad, m, v []float32, lr, beta1, beta2, eps float32, t int) {
	beta1Pow := float32(math.Pow(float64(beta1), float64(t)))
	beta2Pow := float32(math.Pow(float64(beta2), float64(t)))
	denom1 := 1 - beta1Pow
	denom2 := 1 - beta2Pow
	oneMinusBeta1 := 1 - beta1
	oneMinusBeta2 := 1 - beta2

	update := func(start, end int) {
		for i := start; i < end; i++ {
			g := grad[i]
			m[i] = beta1*m[i] + oneMinusBeta1*g
			v[i] = beta2*v[i] + oneMinusBeta2*g*g

			mHat := m[i] / denom1
			vHat := v[i] / denom2

			data[i] -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
		}
	}

	if len(data) > b.parallelThreshold {
		b.pool.Process(len(data), update)
	} else {
		update(0, len(data))
	}
}
