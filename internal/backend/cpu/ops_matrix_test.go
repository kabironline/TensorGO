package test

import (
	"math/rand"
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cpu"
)

func BenchmarkMatMulCPU(b *testing.B) {
	cpu := cpu.NewCPUBackend()
	backend.RegisterBackend("cpu", cpu)

	m := 4096
	k := 2048
	n := 1024

	h_a := make([]float32, m*k)
	h_b := make([]float32, k*n)
	h_c := make([]float32, m*n)

	randomInit(h_a)
	randomInit(h_b)

	// Warmup
	for range [10]struct{}{} {
		cpu.MatMul(
			backend.MatOperand{Data: h_a, Rows: m, Cols: k, LD: k},
			backend.MatOperand{Data: h_b, Rows: k, Cols: n, LD: n},
			h_c,
			1.0, 0.0,
		)
	}

	b.ResetTimer()
	b.SetBytes(int64(m * n * 4 * 3))

	for b.Loop() {
		cpu.MatMul(
			backend.MatOperand{Data: h_a, Rows: m, Cols: k, LD: k},
			backend.MatOperand{Data: h_b, Rows: k, Cols: n, LD: n},
			h_c,
			1.0, 0.0,
		)
	}
}

func randomInit(buf []float32) {
	for i := range buf {
		buf[i] = rand.Float32()
	}
}
