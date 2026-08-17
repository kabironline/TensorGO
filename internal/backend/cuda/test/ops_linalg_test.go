//go:build cuda

package cuda_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/kabironline/nanograd/internal/backend/cuda"
)

func newCUDA(t testing.TB) *cuda.CUDABackend {
	t.Helper()
	devices, err := cuda.GetCudaDeviceCount()
	if err != nil || devices < 1 {
		t.Skipf("no CUDA device available (%v)", err)
	}
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	return cu
}

func identity(n int) []float32 {
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		out[i*n+i] = 1
	}
	return out
}

// wellConditioned builds a diagonally dominant n x n matrix: always invertible
// and stable enough that float32 round-trip error stays predictable.
func wellConditioned(n int) []float32 {
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out[i*n+j] = float32((i*7+j*3)%5) * 0.25
		}
		out[i*n+i] += float32(n)
	}
	return out
}

func assertClose(t *testing.T, got, want []float32, tol float64, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	worst := 0.0
	worstAt := -1
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > worst {
			worst, worstAt = d, i
		}
	}
	if worst > tol {
		t.Fatalf("%s: max deviation %g at element %d (got %v, want %v), tol %g",
			label, worst, worstAt, got[worstAt], want[worstAt], tol)
	}
}

// roundTrip inverts in on the device and returns A @ A^-1, which must be I.
func roundTrip(t *testing.T, cu *cuda.CUDABackend, in []float32, n int) []float32 {
	t.Helper()

	dA := cu.ToDevice(in)
	dInv := cu.Allocate(n * n)
	dProd := cu.Allocate(n * n)
	defer cu.Free(dA)
	defer cu.Free(dInv)
	defer cu.Free(dProd)

	if err := cu.Inverse(dA, dInv, n); err != nil {
		t.Fatalf("n=%d: Inverse returned error: %v", n, err)
	}
	cu.MatMul(
		backend.MatOperand{Data: dA, Rows: n, Cols: n, LD: n},
		backend.MatOperand{Data: dInv, Rows: n, Cols: n, LD: n},
		dProd, 1.0, 0.0,
	)
	cu.Sync()
	return cu.ToCPU(dProd)
}

func TestCudaInverseIdentity(t *testing.T) {
	cu := newCUDA(t)
	const n = 4

	dA := cu.ToDevice(identity(n))
	dOut := cu.Allocate(n * n)
	defer cu.Free(dA)
	defer cu.Free(dOut)

	if err := cu.Inverse(dA, dOut, n); err != nil {
		t.Fatalf("Inverse(I) returned error: %v", err)
	}
	cu.Sync()
	assertClose(t, cu.ToCPU(dOut), identity(n), 1e-5, "inverse of identity")
}

func TestCudaInverseKnown2x2(t *testing.T) {
	cu := newCUDA(t)

	// [[4,7],[2,6]]^-1 = [[0.6,-0.7],[-0.2,0.4]]
	dA := cu.ToDevice([]float32{4, 7, 2, 6})
	dOut := cu.Allocate(4)
	defer cu.Free(dA)
	defer cu.Free(dOut)

	if err := cu.Inverse(dA, dOut, 2); err != nil {
		t.Fatalf("Inverse returned error: %v", err)
	}
	cu.Sync()
	assertClose(t, cu.ToCPU(dOut), []float32{0.6, -0.7, -0.2, 0.4}, 1e-5, "2x2 inverse")
}

// TestCudaInverseRoundTrip straddles the small/large crossover deliberately:
// sizes below, at, and above CUDA_INVERSE_SMALL_MAX exercise both strategies
// through the same public entry point.
func TestCudaInverseRoundTrip(t *testing.T) {
	cu := newCUDA(t)

	smallMax := cuda.InverseSmallMax
	for _, n := range []int{1, 2, 3, 8, smallMax - 1, smallMax, smallMax + 1, 64, 128} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			prod := roundTrip(t, cu, wellConditioned(n), n)
			// Error grows with n: each output element is a length-n dot product.
			assertClose(t, prod, identity(n), 2e-4*float64(n), fmt.Sprintf("A @ A^-1 (n=%d)", n))
		})
	}
}

// TestCudaInversePathsAgree is the point of having two implementations: at a
// size both can handle they must produce the same answer.
func TestCudaInversePathsAgree(t *testing.T) {
	cu := newCUDA(t)

	for _, n := range []int{2, 8, 16, cuda.InverseSmallMax} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			in := wellConditioned(n)

			dA := cu.ToDevice(in)
			dSmall := cu.Allocate(n * n)
			dLarge := cu.Allocate(n * n)
			defer cu.Free(dA)
			defer cu.Free(dSmall)
			defer cu.Free(dLarge)

			if err := cuda.InverseSmallForTest(cu, dA, dSmall, n); err != nil {
				t.Fatalf("small path: %v", err)
			}
			if err := cuda.InverseLargeForTest(cu, dA, dLarge, n); err != nil {
				t.Fatalf("large path: %v", err)
			}
			cu.Sync()

			// Different algorithms (Gauss-Jordan vs blocked LU), so they agree to
			// float32 working precision, not bit-for-bit.
			assertClose(t, cu.ToCPU(dSmall), cu.ToCPU(dLarge), 1e-3,
				fmt.Sprintf("small vs large (n=%d)", n))
		})
	}
}

// TestCudaInverseMatchesCPU pins the GPU result against the gonum-backed CPU
// implementation, which has its own independent test suite.
func TestCudaInverseMatchesCPU(t *testing.T) {
	cu := newCUDA(t)
	host := cpu.NewCPUBackend()

	for _, n := range []int{2, 8, 33, 64} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			in := wellConditioned(n)

			want := make([]float32, n*n)
			if err := host.Inverse(in, want, n); err != nil {
				t.Fatalf("CPU Inverse: %v", err)
			}

			dA := cu.ToDevice(in)
			dOut := cu.Allocate(n * n)
			defer cu.Free(dA)
			defer cu.Free(dOut)

			if err := cu.Inverse(dA, dOut, n); err != nil {
				t.Fatalf("CUDA Inverse: %v", err)
			}
			cu.Sync()
			assertClose(t, cu.ToCPU(dOut), want, 1e-3, fmt.Sprintf("GPU vs CPU (n=%d)", n))
		})
	}
}

func TestCudaInverseSingular(t *testing.T) {
	cu := newCUDA(t)

	// Both paths must report singularity rather than returning garbage.
	t.Run("small", func(t *testing.T) {
		// Second row is 2x the first.
		dA := cu.ToDevice([]float32{1, 2, 2, 4})
		dOut := cu.Allocate(4)
		defer cu.Free(dA)
		defer cu.Free(dOut)

		if err := cu.Inverse(dA, dOut, 2); err == nil {
			t.Fatal("expected an error for a singular matrix, got nil")
		}
	})

	t.Run("large", func(t *testing.T) {
		const n = 64
		in := wellConditioned(n)
		// Make row 5 an exact duplicate of row 4.
		copy(in[5*n:6*n], in[4*n:5*n])

		dA := cu.ToDevice(in)
		dOut := cu.Allocate(n * n)
		defer cu.Free(dA)
		defer cu.Free(dOut)

		if err := cu.Inverse(dA, dOut, n); err == nil {
			t.Fatal("expected an error for a singular matrix, got nil")
		}
	})
}

func TestCudaInverseRejectsBadSizes(t *testing.T) {
	cu := newCUDA(t)
	const n = 4

	dA := cu.ToDevice(wellConditioned(n))
	dOut := cu.Allocate(n * n)
	defer cu.Free(dA)
	defer cu.Free(dOut)

	cases := []struct {
		name string
		a    []float32
		out  []float32
		n    int
	}{
		{"zero n", dA, dOut, 0},
		{"negative n", dA, dOut, -1},
		{"n too large for buffers", dA, dOut, n + 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if err := cu.Inverse(c.a, c.out, c.n); err == nil {
				t.Fatal("expected an error, got nil")
			}
		})
	}
}

// TestCudaInverseDoesNotMutateInput guards the documented contract: getrf
// overwrites its input, so the large path must copy into scratch first.
func TestCudaInverseDoesNotMutateInput(t *testing.T) {
	cu := newCUDA(t)

	for _, n := range []int{8, 64} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			in := wellConditioned(n)
			dA := cu.ToDevice(in)
			dOut := cu.Allocate(n * n)
			defer cu.Free(dA)
			defer cu.Free(dOut)

			if err := cu.Inverse(dA, dOut, n); err != nil {
				t.Fatalf("Inverse: %v", err)
			}
			cu.Sync()
			assertClose(t, cu.ToCPU(dA), in, 0, "input must be unchanged")
		})
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func benchInverse(b *testing.B, n int, path string) {
	cu := newCUDA(b)

	dA := cu.ToDevice(wellConditioned(n))
	dOut := cu.Allocate(n * n)
	defer cu.Free(dA)
	defer cu.Free(dOut)

	run := func() error { return cu.Inverse(dA, dOut, n) }
	switch path {
	case "small":
		run = func() error { return cuda.InverseSmallForTest(cu, dA, dOut, n) }
	case "large":
		run = func() error { return cuda.InverseLargeForTest(cu, dA, dOut, n) }
	}

	// Warm up: first call pays context and cuBLAS workspace setup.
	if err := run(); err != nil {
		b.Fatalf("warmup: %v", err)
	}
	cu.Sync()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := run(); err != nil {
			b.Fatalf("Inverse: %v", err)
		}
	}
	cu.Sync()
}

func BenchmarkCudaInverseAuto8(b *testing.B)   { benchInverse(b, 8, "auto") }
func BenchmarkCudaInverseAuto32(b *testing.B)  { benchInverse(b, 32, "auto") }
func BenchmarkCudaInverseAuto64(b *testing.B)  { benchInverse(b, 64, "auto") }
func BenchmarkCudaInverseAuto256(b *testing.B) { benchInverse(b, 256, "auto") }

// Same n through both strategies -- this is what justifies where the crossover
// sits, and what to re-measure if CUDA_INVERSE_SMALL_MAX is ever changed.
func BenchmarkCudaInverseSmall8(b *testing.B)  { benchInverse(b, 8, "small") }
func BenchmarkCudaInverseLarge8(b *testing.B)  { benchInverse(b, 8, "large") }
func BenchmarkCudaInverseSmall32(b *testing.B) { benchInverse(b, 32, "small") }
func BenchmarkCudaInverseLarge32(b *testing.B) { benchInverse(b, 32, "large") }
