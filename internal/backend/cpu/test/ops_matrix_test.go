package test

import (
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cpu"
)

func BenchmarkMatMulCPU(b *testing.B) {
	cpu := cpu.NewCPUBackend()
	backend.RegisterBackend("cpu", cpu)
	backend.SetDefaultBackend(cpu)

	size := 4096
	h_a := make([]float32, size*size)
	h_b := make([]float32, size*size)
	h_c := make([]float32, size*size)

	// Warmup
	for i := 0; i < 10; i++ {
		cpu.MatMul(
			h_a,
			h_b,
			h_c,
			size, size, size,
			size, size,
		)
	}
	b.ResetTimer()
	b.SetBytes(int64(size * size * 4 * 3))

	// Benchmark
	for i := 0; i < b.N; i++ {
		cpu.MatMul(
			h_a,
			h_b,
			h_c,
			size, size, size,
			size, size,
		)
	}
}
