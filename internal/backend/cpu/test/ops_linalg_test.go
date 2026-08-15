package cpu_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cpu"
)

func identity(n int) []float32 {
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		out[i*n+i] = 1
	}
	return out
}

// wellConditioned builds a diagonally dominant n x n matrix, which is
// guaranteed invertible and numerically stable to invert.
func wellConditioned(n int) []float32 {
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out[i*n+j] = float32((i*7+j*3)%5) * 0.25
		}
		out[i*n+i] += float32(n) // dominate the diagonal
	}
	return out
}

func assertClose(t *testing.T, got, want []float32, tol float64, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s: element %d = %v, want %v\n got:  %v\n want: %v",
				label, i, got[i], want[i], got, want)
		}
	}
}

// didPanic reports whether fn panicked, so a function that is supposed to
// return an error but panics instead fails cleanly rather than aborting the run.
func didPanic(fn func()) (panicked bool, value any) {
	defer func() {
		if r := recover(); r != nil {
			panicked, value = true, r
		}
	}()
	fn()
	return false, nil
}

func TestInverseIdentity(t *testing.T) {
	bk := cpu.NewCPUBackend()
	const n = 4

	out := make([]float32, n*n)
	if err := bk.Inverse(identity(n), out, n); err != nil {
		t.Fatalf("Inverse(I) returned error: %v", err)
	}
	assertClose(t, out, identity(n), 1e-5, "inverse of identity")
}

func TestInverseKnown2x2(t *testing.T) {
	bk := cpu.NewCPUBackend()

	// [[4,7],[2,6]]^-1 = [[0.6,-0.7],[-0.2,0.4]]
	out := make([]float32, 4)
	if err := bk.Inverse([]float32{4, 7, 2, 6}, out, 2); err != nil {
		t.Fatalf("Inverse returned error: %v", err)
	}
	assertClose(t, out, []float32{0.6, -0.7, -0.2, 0.4}, 1e-5, "2x2 inverse")
}

// TestInverseRoundTrip is the general check: A @ A^-1 must be the identity for
// any invertible A, at any size, with no hand-computed answer needed.
func TestInverseRoundTrip(t *testing.T) {
	bk := cpu.NewCPUBackend()

	for _, n := range []int{1, 2, 3, 8, 16} {
		in := wellConditioned(n)
		inv := make([]float32, n*n)

		if err := bk.Inverse(in, inv, n); err != nil {
			t.Fatalf("n=%d: Inverse returned error: %v", n, err)
		}

		prod := make([]float32, n*n)
		bk.MatMul(
			backend.MatOperand{Data: in, Rows: n, Cols: n, LD: n},
			backend.MatOperand{Data: inv, Rows: n, Cols: n, LD: n},
			prod, 1.0, 0.0,
		)

		// Tolerance scales with n: more terms accumulate more float32 error.
		assertClose(t, prod, identity(n), 1e-4*float64(n),
			fmt.Sprintf("A @ A^-1 (n=%d)", n))
	}
}

// TestInverseDoesNotMutateInput guards the no-alias contract from the other
// side: writing out must not disturb a.
func TestInverseDoesNotMutateInput(t *testing.T) {
	bk := cpu.NewCPUBackend()
	const n = 3

	in := wellConditioned(n)
	orig := append([]float32(nil), in...)

	if err := bk.Inverse(in, make([]float32, n*n), n); err != nil {
		t.Fatalf("Inverse returned error: %v", err)
	}
	assertClose(t, in, orig, 0, "input must be unchanged")
}

func TestInverseSingularReturnsError(t *testing.T) {
	bk := cpu.NewCPUBackend()

	// Second row is 2x the first -> determinant 0.
	var err error
	panicked, val := didPanic(func() {
		err = bk.Inverse([]float32{1, 2, 2, 4}, make([]float32, 4), 2)
	})

	if panicked {
		t.Fatalf("Inverse panicked on a singular matrix instead of returning an error: %v\n"+
			"singularity is a property of the data, not a programmer mistake -- it "+
			"must not take down the process", val)
	}
	if err == nil {
		t.Fatal("Inverse returned nil error for a singular matrix")
	}
}

// Size validation: a short input or a short output must be reported, never
// silently truncated. copy() would happily do the wrong thing here.
func TestInverseRejectsBadSizes(t *testing.T) {
	bk := cpu.NewCPUBackend()
	const n = 3

	cases := []struct {
		name string
		a    []float32
		out  []float32
		n    int
	}{
		{"short input", make([]float32, n*n-1), make([]float32, n*n), n},
		{"short output", wellConditioned(n), make([]float32, n*n-1), n},
		{"zero n", wellConditioned(n), make([]float32, n*n), 0},
		{"negative n", wellConditioned(n), make([]float32, n*n), -1},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var err error
			panicked, val := didPanic(func() { err = bk.Inverse(c.a, c.out, c.n) })
			if panicked {
				t.Fatalf("panicked instead of returning an error: %v", val)
			}
			if err == nil {
				t.Fatal("returned nil error")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func benchmarkInverse(b *testing.B, n int) {
	bk := cpu.NewCPUBackend()
	in := wellConditioned(n)
	out := make([]float32, n*n)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := bk.Inverse(in, out, n); err != nil {
			b.Fatalf("Inverse returned error: %v", err)
		}
	}
}

func BenchmarkInverse8(b *testing.B)   { benchmarkInverse(b, 8) }
func BenchmarkInverse64(b *testing.B)  { benchmarkInverse(b, 64) }
func BenchmarkInverse256(b *testing.B) { benchmarkInverse(b, 256) }
