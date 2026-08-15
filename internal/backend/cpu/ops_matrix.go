package cpu

import (
	"fmt"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/mat"

	"github.com/kabironline/nanograd/backend"
)

// Matrix operations for the CPU backend.

// general builds the gonum descriptor for op. gonum wants the *stored* shape
// plus a separate transpose flag, which is exactly what MatOperand carries.
func general(op backend.MatOperand) (blas32.General, blas.Transpose) {
	rows, cols := op.StoredDims()
	t := blas.NoTrans
	if op.Trans {
		t = blas.Trans
	}
	return blas32.General{
		Rows:   rows,
		Cols:   cols,
		Stride: op.LD,
		Data:   op.Data,
	}, t
}

// MatMul computes out = alpha*(A @ B) + beta*out.
func (bk *CPUBackend) MatMul(a, b backend.MatOperand, out []float32, alpha, beta float32) {
	ga, ta := general(a)
	gb, tb := general(b)

	blas32.Gemm(
		ta, tb,
		alpha,
		ga,
		gb,
		beta,
		blas32.General{
			Rows:   a.Rows,
			Cols:   b.Cols,
			Stride: b.Cols,
			Data:   out,
		},
	)
}

func (bk *CPUBackend) Inverse(a []float32, out []float32, n int) error {
	if n <= 0 {
		return fmt.Errorf("Inverse: n must be positive, got %d", n)
	}
	if len(a) < n*n {
		return fmt.Errorf("Inverse: input has %d elements, need %d", len(a), n*n)
	}
	if len(out) < n*n {
		return fmt.Errorf("Inverse: out has %d elements, need %d", len(out), n*n)
	}

	aF64 := make([]float64, n*n)
	for i := range aF64 {
		aF64[i] = float64(a[i])
	}

	var inv mat.Dense
	if err := inv.Inverse(mat.NewDense(n, n, aF64)); err != nil {
		return fmt.Errorf("Inverse: %w", err)
	}

	for i, v := range inv.RawMatrix().Data { // straight into out
		out[i] = float32(v)
	}
	return nil
}
