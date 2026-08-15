package tensor_test

import (
	"math"
	"testing"

	_ "github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/kabironline/nanograd/tensor"
)

func newTestTensor(data []float32, shape []int) *tensor.Tensor {
	t := tensor.NewTensor(data, shape)
	t.RequiresGrad = true
	return t
}

func TestBackPropSimpleAdd(t *testing.T) {
	x := newTestTensor([]float32{2.0}, []int{1})
	y := newTestTensor([]float32{3.0}, []int{1})
	z := x.Add(y)

	z.BackProp()

	if x.Grad()[0] != 1.0 {
		t.Errorf("Expected x.Grad() to be 1.0, got %f", x.Grad()[0])
	}
	if y.Grad()[0] != 1.0 {
		t.Errorf("Expected y.Grad() to be 1.0, got %f", y.Grad()[0])
	}
}

func TestBackPropAccumulation(t *testing.T) {
	x := newTestTensor([]float32{2.0}, []int{1})
	// z = x + x
	z := x.Add(x)

	z.BackProp()

	// If your engine uses "=" instead of "+=" in the Backward closure,
	// this will wrongly return 1.0 instead of 2.0.
	if x.Grad()[0] != 2.0 {
		t.Errorf("Expected x.Grad() to be 2.0 (accumulation), got %f", x.Grad()[0])
	}
}

func TestBackPropChainRule(t *testing.T) {
	x := newTestTensor([]float32{2.0}, []int{1})
	y := newTestTensor([]float32{3.0}, []int{1})

	// z = x * y + y
	mul := x.Mul(y)
	z := mul.Add(y)

	z.BackProp()

	// dz/dx = y = 3.0
	if x.Grad()[0] != 3.0 {
		t.Errorf("Chain Rule Fail: Expected x.Grad() 3.0, got %f", x.Grad()[0])
	}
	// dz/dy = x + 1 = 2.0 + 1.0 = 3.0
	if y.Grad()[0] != 3.0 {
		t.Errorf("Chain Rule Fail: Expected y.Grad() 3.0, got %f", y.Grad()[0])
	}
}

func TestBackPropSimpleMul(t *testing.T) {
	x := newTestTensor([]float32{2.0}, []int{1})
	y := newTestTensor([]float32{3.0}, []int{1})
	z := x.Mul(y)

	z.BackProp()

	if x.Grad()[0] != 3.0 {
		t.Errorf("Expected x.Grad() to be 3.0, got %f", x.Grad()[0])
	}
	if y.Grad()[0] != 2.0 {
		t.Errorf("Expected y.Grad() to be 2.0, got %f", y.Grad()[0])
	}
}

func TestBackPropMulWithTransposedView(t *testing.T) {
	a := newTestTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	bBase := newTestTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	b := bBase.Transpose([]int{1, 0})

	prod := a.Mul(b)
	loss := prod.Sum()
	loss.BackProp()

	expectedAGrad := []float32{5, 7, 6, 8}
	for i, v := range expectedAGrad {
		if math.Abs(float64(a.Grad()[i]-v)) > 1e-9 {
			t.Errorf("a.Grad()[%d]: expected %f, got %f", i, v, a.Grad()[i])
		}
	}

	if bBase.Grad() == nil {
		t.Fatalf("Expected bBase.Grad() to be allocated")
	}

	expectedBBaseGrad := []float32{1, 3, 2, 4}
	for i, v := range expectedBBaseGrad {
		if math.Abs(float64(bBase.Grad()[i]-v)) > 1e-9 {
			t.Errorf("bBase.Grad()[%d]: expected %f, got %f", i, v, bBase.Grad()[i])
		}
	}
}

func TestBackPropChain(t *testing.T) {
	// z = (x + y) * w
	// dz/dx = w
	// dz/dy = w
	// dz/dw = x + y
	x := newTestTensor([]float32{2.0}, []int{1})
	y := newTestTensor([]float32{3.0}, []int{1})
	w := newTestTensor([]float32{4.0}, []int{1})

	sum := x.Add(y)
	z := sum.Mul(w)

	z.BackProp()

	if x.Grad()[0] != 4.0 {
		t.Errorf("Expected x.Grad() to be 4.0, got %f", x.Grad()[0])
	}
	if y.Grad()[0] != 4.0 {
		t.Errorf("Expected y.Grad() to be 4.0, got %f", y.Grad()[0])
	}
	if w.Grad()[0] != 5.0 {
		t.Errorf("Expected w.Grad() to be 5.0, got %f", w.Grad()[0])
	}
}

func TestBackPropMatMul(t *testing.T) {
	// A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
	// C = A * B = [[19, 22], [43, 50]]
	// dC/dA = GradOut * B^T
	// dC/dB = A^T * GradOut
	// If GradOut = [[1, 1], [1, 1]]
	// dC/dA = [[1, 1], [1, 1]] * [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
	// dC/dB = [[1, 3], [2, 4]] * [[1, 1], [1, 1]] = [[4, 4], [6, 6]]

	a := newTestTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	b := newTestTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	c := a.MatMul(b)

	c.BackProp()

	expectedGradA := []float32{11, 15, 11, 15}
	expectedGradB := []float32{4, 4, 6, 6}

	for i, v := range expectedGradA {
		if math.Abs(float64(a.Grad()[i]-v)) > 1e-9 {
			t.Errorf("At index %d: expected a.Grad() %f, got %f", i, v, a.Grad()[i])
		}
	}
	for i, v := range expectedGradB {
		if math.Abs(float64(b.Grad()[i]-v)) > 1e-9 {
			t.Errorf("At index %d: expected b.Grad() %f, got %f", i, v, b.Grad()[i])
		}
	}
}

func TestBackPropBroadcastAdd(t *testing.T) {
	// x: [2, 2] = [[1, 2], [3, 4]]
	// y: [1, 2] = [[10, 20]]
	// z = x + y = [[11, 22], [13, 24]]
	// dz/dx = [[1, 1], [1, 1]]
	// dz/dy = [[1, 1], [1, 1]] summed over dim 0 -> [2, 2]
	x := newTestTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	y := newTestTensor([]float32{10, 20}, []int{1, 2})
	z := x.Add(y)

	z.BackProp()

	expectedGradX := []float32{1, 1, 1, 1}
	expectedGradY := []float32{2, 2}

	for i, v := range expectedGradX {
		if math.Abs(float64(x.Grad()[i]-v)) > 1e-9 {
			t.Errorf("At index %d: expected x.Grad() %f, got %f", i, v, x.Grad()[i])
		}
	}
	for i, v := range expectedGradY {
		if math.Abs(float64(y.Grad()[i]-v)) > 1e-9 {
			t.Errorf("At index %d: expected y.Grad() %f, got %f", i, v, y.Grad()[i])
		}
	}
}

func TestBackPropSum(t *testing.T) {
	x := newTestTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	z := x.Sum()

	z.BackProp()

	for i := range x.Grad() {
		if x.Grad()[i] != 1.0 {
			t.Errorf("Expected x.Grad()[%d] to be 1.0, got %f", i, x.Grad()[i])
		}
	}
}

func TestBackPropMean(t *testing.T) {
	x := newTestTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	z := x.Mean()

	z.BackProp()

	for i := range x.Grad() {
		if x.Grad()[i] != 0.25 {
			t.Errorf("Expected x.Grad()[%d] to be 0.25, got %f", i, x.Grad()[i])
		}
	}
}

func TestBackPropComplex(t *testing.T) {
	// L = sum((A * B + C) * D)
	// A: [2, 2], B: [2, 2], C: [2, 2], D: [2, 2]
	a := newTestTensor([]float32{1, 0, 0, 1}, []int{2, 2})
	b := newTestTensor([]float32{2, 3, 4, 5}, []int{2, 2})
	c := newTestTensor([]float32{1, 1, 1, 1}, []int{2, 2})
	d := newTestTensor([]float32{0.5, 0.5, 0.5, 0.5}, []int{2, 2})

	mm := a.MatMul(b) // [[2, 3], [4, 5]]
	add := mm.Add(c)  // [[3, 4], [5, 6]]
	mul := add.Mul(d) // [[1.5, 2], [2.5, 3]]
	loss := mul.Sum() // 9.0

	loss.BackProp()

	// dL/dmul = [[1, 1], [1, 1]]
	// dL/dd = add = [[3, 4], [5, 6]]
	// dL/dadd = dL/dmul * d = [[0.5, 0.5], [0.5, 0.5]]
	// dL/dc = dL/dadd = [[0.5, 0.5], [0.5, 0.5]]
	// dL/dmm = dL/dadd = [[0.5, 0.5], [0.5, 0.5]]
	// dL/da = dL/dmm * B^T = [[0.5, 0.5], [0.5, 0.5]] * [[2, 4], [3, 5]] = [[2.5, 4.5], [2.5, 4.5]]
	// dL/db = A^T * dL/dmm = [[1, 0], [0, 1]] * [[0.5, 0.5], [0.5, 0.5]] = [[0.5, 0.5], [0.5, 0.5]]

	expectedGradA := []float32{2.5, 4.5, 2.5, 4.5}
	expectedGradB := []float32{0.5, 0.5, 0.5, 0.5}
	expectedGradC := []float32{0.5, 0.5, 0.5, 0.5}
	expectedGradD := []float32{3, 4, 5, 6}

	for i, v := range expectedGradA {
		if math.Abs(float64(a.Grad()[i]-v)) > 1e-9 {
			t.Errorf("a.Grad()[%d]: expected %f, got %f", i, v, a.Grad()[i])
		}
	}
	for i, v := range expectedGradB {
		if math.Abs(float64(b.Grad()[i]-v)) > 1e-9 {
			t.Errorf("b.Grad()[%d]: expected %f, got %f", i, v, b.Grad()[i])
		}
	}
	for i, v := range expectedGradC {
		if math.Abs(float64(c.Grad()[i]-v)) > 1e-9 {
			t.Errorf("c.Grad()[%d]: expected %f, got %f", i, v, c.Grad()[i])
		}
	}
	for i, v := range expectedGradD {
		if math.Abs(float64(d.Grad()[i]-v)) > 1e-9 {
			t.Errorf("d.Grad()[%d]: expected %f, got %f", i, v, d.Grad()[i])
		}
	}
}

func BenchmarkBackPropChain(b *testing.B) {
	x := tensor.NewTensor([]float32{2.0}, []int{1})
	y := tensor.NewTensor([]float32{3.0}, []int{1})
	w := tensor.NewTensor([]float32{4.0}, []int{1})

	for b.Loop() {
		sum := x.Add(y)
		z := sum.Mul(w)
		z.BackProp()
		// Reset gradients for next iteration
		x.Grad()[0], y.Grad()[0], w.Grad()[0] = 0.0, 0.0, 0.0
	}
}

func BenchmarkBackPropMatMul(b *testing.B) {
	a := tensor.NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	bb := tensor.NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})

	for b.Loop() {
		c := a.MatMul(bb)
		c.BackProp()
		// Reset gradients for next iteration
		for i := range a.Grad() {
			a.Grad()[i] = 0.0
		}
		for i := range bb.Grad() {
			bb.Grad()[i] = 0.0
		}
	}
}

func BenchmarkComplexBackProp(b *testing.B) {
	a := newTestTensor([]float32{1, 0, 0, 1}, []int{2, 2})
	bb := newTestTensor([]float32{2, 3, 4, 5}, []int{2, 2})
	c := newTestTensor([]float32{1, 1, 1, 1}, []int{2, 2})
	d := newTestTensor([]float32{0.5, 0.5, 0.5, 0.5}, []int{2, 2})

	for b.Loop() {
		// mm = (a * b + c) * d
		mm := a.MatMul(bb)
		add := mm.Add(c)
		mul := add.Mul(d)
		loss := mul.Sum()
		loss.BackProp()
		// Reset gradients for next iteration

		for i := range a.Grad() {
			a.Grad()[i] = 0.0
		}
		for i := range bb.Grad() {
			bb.Grad()[i] = 0.0
		}
		for i := range c.Grad() {
			c.Grad()[i] = 0.0
		}
		for i := range d.Grad() {
			d.Grad()[i] = 0.0
		}
	}
}
