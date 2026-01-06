package cpu

import "math"

func (c *CPUBackend) Exp(x []float64, n int) []float64 {
	res := make([]float64, n)
	c.pool.Process(n, func(start, end int) {
		for i := start; i < end; i++ {
			res[i] = math.Exp(x[i])
		}
	})
	return res
}

func (c *CPUBackend) Log(x []float64, n int) []float64 {
	res := make([]float64, n)
	c.pool.Process(n, func(start, end int) {
		for i := start; i < end; i++ {
			res[i] = math.Log(x[i])
		}
	})
	return res
}

func (c *CPUBackend) Square(x []float64, n int) []float64 {
	res := make([]float64, n)
	c.pool.Process(n, func(start, end int) {
		for i := start; i < end; i++ {
			res[i] = x[i] * x[i]
		}
	})
	return res
}
