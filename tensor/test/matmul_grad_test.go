package tensor_test

import (
	"math"
	"testing"

	"github.com/kabironline/nanograd/tensor"
)

func closeEnough(t *testing.T, got, want []float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d (got %v)", label, len(got), len(want), got)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("%s: got %v, want %v", label, got, want)
		}
	}
}

// Gradients must flow through a transposed operand.
func TestMatMulTransposedBackward(t *testing.T) {
	a := tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	a.RequiresGrad = true
	b := tensor.NewTensor([]float32{7, 8, 9, 10, 11, 12}, []int{2, 3})
	b.RequiresGrad = true

	// out = a @ b^T  -> (2,2); seed dL/dout = ones.
	out := a.MatMul(b.Transpose([]int{1, 0}))
	out.Sum().BackProp()

	// With dL/dout all ones: dL/da = ones(2,2) @ b_T^T = column sums of b broadcast.
	// b^T is (3,2) so dL/da (2,3) rows are both [7+10, 8+11, 9+12] = [17,19,21].
	closeEnough(t, a.Grad(), []float32{17, 19, 21, 17, 19, 21}, "grad a")

	// dL/db (2,3): both rows are column sums of a = [1+4, 2+5, 3+6] = [5,7,9].
	closeEnough(t, b.Grad(), []float32{5, 7, 9, 5, 7, 9}, "grad b")
}

// MatVecMul must deliver a gradient to the vector (previously dropped).
func TestMatVecMulVectorGradient(t *testing.T) {
	m := tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	m.RequiresGrad = true
	v := tensor.NewTensor([]float32{1, 1, 1}, []int{3})
	v.RequiresGrad = true

	m.MatVecMul(v).Sum().BackProp()

	if v.Grad() == nil {
		t.Fatal("MatVecMul: vector gradient is nil — gradient was dropped")
	}
	// dL/dv = column sums of m = [1+4, 2+5, 3+6]
	closeEnough(t, v.Grad(), []float32{5, 7, 9}, "grad v")
}
