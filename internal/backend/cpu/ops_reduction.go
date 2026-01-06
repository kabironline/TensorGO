package cpu

// Sum computes the sum of all elements
func (bk *CPUBackend) Sum(a []float64, size int) float64 {
	var sum float64
	for i := range size {
		sum += a[i]
	}
	return sum
}

// Mean computes the mean of all elements
func (bk *CPUBackend) Mean(a []float64, size int) float64 {
	return bk.Sum(a, size) / float64(size)
}

// Max finds the maximum element
func (bk *CPUBackend) Max(a []float64, size int) float64 {
	maxVal := a[0]
	for i := 1; i < size; i++ {
		if a[i] > maxVal {
			maxVal = a[i]
		}
	}
	return maxVal
}

// Min finds the minimum element
func (bk *CPUBackend) Min(a []float64, size int) float64 {
	minVal := a[0]
	for i := 1; i < size; i++ {
		if a[i] < minVal {
			minVal = a[i]
		}
	}
	return minVal
}
