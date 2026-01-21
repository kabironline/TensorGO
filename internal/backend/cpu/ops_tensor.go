package cpu

// CPU implementation of tensor-related ops (Contiguous, etc.)

// Contiguous writes a contiguous copy of `data` (interpreted with `shape` and
// `strides`) into `outData`. `data` is expected to be already offset such that
// its logical origin corresponds to index 0. If `outData` is nil or too small,
// this function will allocate and return a buffer; however callers are expected
// to pre-allocate an appropriate output buffer (see tensor.Contiguous).
func (b *CPUBackend) Contiguous(data, outData []float32, shape, strides []int, offset int) {
	// Compute size
	size := 1
	for _, d := range shape {
		size *= d
	}

	// Ensure outData is large enough
	if outData == nil || len(outData) < size {
		outData = b.Allocate(size)
	}

	// taking offset into account
	data = data[offset:]

	// Map each logical linear index to the physical index in `data` using the
	// provided `strides` and copy into the contiguous output.
	for i := 0; i < size; i++ {
		physical := 0
		idx := i
		for d := len(shape) - 1; d >= 0; d-- {
			physical += (idx % shape[d]) * strides[d]
			idx /= shape[d]
		}
		outData[i] = data[physical]
	}
}
