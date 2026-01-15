package cpu

import "math"

func (c *CPUBackend) Exp(x []float32, n int) []float32 {
	res := make([]float32, n)
	c.pool.Process(n, func(start, end int) {
		for i := start; i < end; i++ {
			res[i] = float32(math.Exp(float64(x[i])))
		}
	})
	return res
}

func (c *CPUBackend) Log(x []float32, n int) []float32 {
	res := make([]float32, n)
	c.pool.Process(n, func(start, end int) {
		for i := start; i < end; i++ {
			res[i] = float32(math.Log(float64(x[i])))
		}
	})
	return res
}

func (c *CPUBackend) Square(x []float32, n int) []float32 {
	res := make([]float32, n)
	c.pool.Process(n, func(start, end int) {
		for i := start; i < end; i++ {
			res[i] = x[i] * x[i]
		}
	})
	return res
}
