package cpu

import (
	"math/rand"
)

func (bk *CPUBackend) Normal(data []float32, mean, stdDev float32, size int) {
	for i := 0; i < size; i++ {
		data[i] = mean + float32(rand.NormFloat64())*stdDev
	}
}
