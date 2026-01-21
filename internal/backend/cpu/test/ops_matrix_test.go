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

	m := 4096
	k := 2048
	n := 1024
	h_a := make([]float32, m*k)
	h_b := make([]float32, k*n)
	h_c := make([]float32, m*n)

	// Warmup
	for range [10]struct{}{} {
		cpu.MatMul(
			h_a,
			h_b,
			h_c,
			m, n, k,
			// stride a, b
			k, n,
		)
	}
	b.ResetTimer()
	b.SetBytes(int64(m * n * 4 * 3))

	for b.Loop() {
		cpu.MatMul(
			h_a,
			h_b,
			h_c,
			m, n, k,
			// stride a, b
			k, n,
		)
	}
}
