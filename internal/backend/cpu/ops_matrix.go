package cpu

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"

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
