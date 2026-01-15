package cpu

func (c *CPUBackend) Fill(data []float32, value float32, size int) {
	for i := 0; i < size; i++ {
		data[i] = value
	}
}

func (bk *CPUBackend) Clone(data []float32, size int) []float32 {
	out := bk.Allocate(size)
	copy(out, data[:size])
	return out
}

func (bk *CPUBackend) Transpose(a []float32, rows, cols int) []float32 {
	out := bk.Allocate(rows * cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = a[r*cols+c]
		}
	}
	return out
}
