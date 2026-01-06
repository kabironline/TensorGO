package cpu

import (
	"math/rand"
)

func (bk *CPUBackend) Normal(data []float64, mean, stdDev float64, size int) {
	for i := 0; i < size; i++ {
		data[i] = mean + rand.NormFloat64()*stdDev
	}
}
