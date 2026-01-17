package test

import (
	"testing"

	"github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/stretchr/testify/assert"
)

func TestCPUAdd(t *testing.T) {
	bk := cpu.NewCPUBackend()
	size := 1024
	a := make([]float32, size)
	b := make([]float32, size)
	out := make([]float32, size)
	for i := 0; i < size; i++ {
		a[i] = float32(i)
		b[i] = float32(2.0)
	}

	bk.Add(a, b, out, size)
	for i := 0; i < size; i++ {
		assert.Equal(t, a[i]+b[i], out[i])
	}
}

func BenchmarkCPUAdd(b *testing.B) {
	bk := cpu.NewCPUBackend()
	size := 1 << 20 // 1M elements
	a := make([]float32, size)
	bvec := make([]float32, size)
	out := make([]float32, size)
	for i := 0; i < size; i++ {
		a[i] = 1.0
		bvec[i] = 2.0
	}

	// Warmup
	for i := 0; i < 10; i++ {
		bk.Add(a, bvec, out, size)
	}

	b.ResetTimer()
	b.SetBytes(int64(size * 4 * 3))

	for i := 0; i < b.N; i++ {
		bk.Add(a, bvec, out, size)
	}
}
