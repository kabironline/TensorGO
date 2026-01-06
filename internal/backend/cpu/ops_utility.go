package cpu

func (c *CPUBackend) Fill(data []float64, value float64, size int) {
	for i := 0; i < size; i++ {
		data[i] = value
	}
}

func (bk *CPUBackend) Clone(data []float64, size int) []float64 {
	out := bk.Allocate(size)
	copy(out, data[:size])
	return out
}

func (bk *CPUBackend) Transpose(a []float64, rows, cols int) []float64 {
	out := bk.Allocate(rows * cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = a[r*cols+c]
		}
	}
	return out
}
