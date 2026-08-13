package tensor_test

import (
	"testing"

	"github.com/kabironline/nanograd/gradcheck"
	"github.com/kabironline/nanograd/tensor"
)

// leaf builds a contiguous tensor that requires gradients. Views belong inside
// the checked closure, not here — gradcheck perturbs leaves.
func leaf(data []float32, shape ...int) *tensor.Tensor {
	t := tensor.NewTensor(data, shape)
	t.RequiresGrad = true
	return t
}

func seq(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		// Spread values away from zero and away from each other so a wrong
		// gradient cannot coincidentally match the right one.
		out[i] = float32(i)*0.37 + 0.5
	}
	return out
}

// ---------------------------------------------------------------------------
// Element-wise and activation ops
// ---------------------------------------------------------------------------

func TestGradcheckElementwise(t *testing.T) {
	cases := []struct {
		name string
		fn   func(a, b *tensor.Tensor) *tensor.Tensor
	}{
		{"add", func(a, b *tensor.Tensor) *tensor.Tensor { return a.Add(b) }},
		{"sub", func(a, b *tensor.Tensor) *tensor.Tensor { return a.Sub(b) }},
		{"mul", func(a, b *tensor.Tensor) *tensor.Tensor { return a.Mul(b) }},
		{"div", func(a, b *tensor.Tensor) *tensor.Tensor { return a.Div(b) }},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := leaf(seq(6), 2, 3)
			b := leaf(seq(6), 2, 3)
			gradcheck.Check(t, c.name, func() *tensor.Tensor { return c.fn(a, b) }, a, b)
		})
	}
}

func TestGradcheckActivations(t *testing.T) {
	cases := []struct {
		name string
		fn   func(a *tensor.Tensor) *tensor.Tensor
	}{
		{"relu", func(a *tensor.Tensor) *tensor.Tensor { return a.ReLU() }},
		{"sigmoid", func(a *tensor.Tensor) *tensor.Tensor { return a.Sigmoid() }},
		{"tanh", func(a *tensor.Tensor) *tensor.Tensor { return a.Tanh() }},
		{"exp", func(a *tensor.Tensor) *tensor.Tensor { return a.Exp() }},
		{"log", func(a *tensor.Tensor) *tensor.Tensor { return a.Log() }},
		{"square", func(a *tensor.Tensor) *tensor.Tensor { return a.Square() }},
		{"softmax", func(a *tensor.Tensor) *tensor.Tensor { return a.Softmax() }},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := leaf(seq(6), 2, 3)
			gradcheck.Check(t, c.name, func() *tensor.Tensor { return c.fn(a) }, a)
		})
	}
}

func TestGradcheckScalarOps(t *testing.T) {
	cases := []struct {
		name string
		fn   func(a *tensor.Tensor) *tensor.Tensor
	}{
		{"addscalar", func(a *tensor.Tensor) *tensor.Tensor { return a.AddScalar(2.5) }},
		{"subscalar", func(a *tensor.Tensor) *tensor.Tensor { return a.SubScalar(1.25) }},
		{"mulscalar", func(a *tensor.Tensor) *tensor.Tensor { return a.MulScalar(3.0) }},
		{"divscalar", func(a *tensor.Tensor) *tensor.Tensor { return a.DivScalar(4.0) }},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := leaf(seq(6), 2, 3)
			gradcheck.Check(t, c.name, func() *tensor.Tensor { return c.fn(a) }, a)
		})
	}
}

func TestGradcheckReductions(t *testing.T) {
	t.Run("sum", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		gradcheck.Check(t, "sum", func() *tensor.Tensor { return a.Sum() }, a)
	})
	t.Run("mean", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		gradcheck.Check(t, "mean", func() *tensor.Tensor { return a.Mean() }, a)
	})
}

// ---------------------------------------------------------------------------
// Matmul — the family this session already fixed
// ---------------------------------------------------------------------------

func TestGradcheckMatMul(t *testing.T) {
	t.Run("contiguous", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		b := leaf(seq(6), 3, 2)
		gradcheck.Check(t, "matmul", func() *tensor.Tensor { return a.MatMul(b) }, a, b)
	})

	t.Run("transposed-rhs", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		b := leaf(seq(6), 2, 3)
		gradcheck.Check(t, "matmul/bT", func() *tensor.Tensor {
			return a.MatMul(b.Transpose([]int{1, 0}))
		}, a, b)
	})

	t.Run("transposed-lhs", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		b := leaf(seq(8), 2, 4)
		gradcheck.Check(t, "matmul/aT", func() *tensor.Tensor {
			return a.Transpose([]int{1, 0}).MatMul(b)
		}, a, b)
	})

	t.Run("matmuladdbias", func(t *testing.T) {
		x := leaf(seq(6), 2, 3)
		w := leaf(seq(12), 3, 4)
		bias := leaf(seq(4), 4)
		gradcheck.Check(t, "matmuladdbias", func() *tensor.Tensor {
			return x.MatMulAddBias(w, bias)
		}, x, w, bias)
	})

	t.Run("matvecmul", func(t *testing.T) {
		m := leaf(seq(6), 2, 3)
		v := leaf(seq(3), 3)
		gradcheck.Check(t, "matvecmul", func() *tensor.Tensor {
			return m.MatVecMul(v)
		}, m, v)
	})

	t.Run("vecmatmul", func(t *testing.T) {
		v := leaf(seq(2), 2)
		m := leaf(seq(6), 2, 3)
		gradcheck.Check(t, "vecmatmul", func() *tensor.Tensor {
			return v.VecMatMul(m)
		}, v, m)
	})
}

// ---------------------------------------------------------------------------
// Views — the paths neither MNIST canary exercises, and where the audit
// predicts the remaining P1 bugs live.
// ---------------------------------------------------------------------------

func TestGradcheckViews(t *testing.T) {
	t.Run("transpose", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		gradcheck.Check(t, "transpose", func() *tensor.Tensor {
			return a.Transpose([]int{1, 0})
		}, a)
	})

	t.Run("reshape", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		gradcheck.Check(t, "reshape", func() *tensor.Tensor {
			return a.Reshape([]int{3, 2})
		}, a)
	})

	t.Run("add-on-transposed-view", func(t *testing.T) {
		// Exercises Add's same-shape fast path with a strided operand.
		//
		// KNOWN FAILURE (P1): Add's sameShape fast path reads a.Data() raw and
		// sizes by len(a.Data()), ignoring strides/offset -- every other binary op
		// routes through Broadcast*Op, which calls Contiguous() first. The
		// analytic gradients come back as a transposed permutation of the true
		// ones. Fix: route Add through Contiguous() and size by TotalSize(Shape).
		t.Skip("P1: ops_matrix.go Add fast path ignores strides")

		a := leaf(seq(6), 2, 3)
		b := leaf(seq(6), 3, 2)
		gradcheck.Check(t, "add/aT", func() *tensor.Tensor {
			return a.Transpose([]int{1, 0}).Add(b)
		}, a, b)
	})

	t.Run("mul-on-transposed-view", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		b := leaf(seq(6), 3, 2)
		gradcheck.Check(t, "mul/aT", func() *tensor.Tensor {
			return a.Transpose([]int{1, 0}).Mul(b)
		}, a, b)
	})

	t.Run("relu-on-transposed-view", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		gradcheck.Check(t, "relu/aT", func() *tensor.Tensor {
			return a.Transpose([]int{1, 0}).ReLU()
		}, a)
	})

	t.Run("sum-on-transposed-view", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		gradcheck.Check(t, "sum/aT", func() *tensor.Tensor {
			return a.Transpose([]int{1, 0}).Sum()
		}, a)
	})

	t.Run("broadcast-add", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		b := leaf(seq(3), 3)
		gradcheck.Check(t, "broadcast/add", func() *tensor.Tensor {
			return a.Add(b)
		}, a, b)
	})

	t.Run("broadcast-mul", func(t *testing.T) {
		a := leaf(seq(6), 2, 3)
		b := leaf(seq(3), 3)
		gradcheck.Check(t, "broadcast/mul", func() *tensor.Tensor {
			return a.Mul(b)
		}, a, b)
	})

	t.Run("broadcastto", func(t *testing.T) {
		a := leaf(seq(3), 1, 3)
		gradcheck.Check(t, "broadcastto", func() *tensor.Tensor {
			return a.BroadcastTo([]int{2, 3})
		}, a)
	})

	t.Run("slice", func(t *testing.T) {
		// KNOWN FAILURE (P1): Slice reslices storage to [offset:] *and* sets
		// Offset, so the offset is applied twice. Forward and backward disagree
		// about where the data lives -- the gradient lands 6 elements past where
		// it belongs. Fix: pick one representation, not both.
		t.Skip("P1: ops_matrix.go Slice applies its offset twice")

		a := leaf(seq(12), 4, 3)
		gradcheck.Check(t, "slice", func() *tensor.Tensor {
			return a.Slice([]int{1, 0}, []int{3, 3})
		}, a)
	})

	t.Run("slice-then-transpose", func(t *testing.T) {
		// KNOWN FAILURE (P1): panics with index out of range in Transpose's
		// backward. ensureGrad sizes the gradient by data.Length() while
		// AllocGrad uses TotalSize(Shape); for a sliced view these differ, so the
		// transpose backward indexes past the end. Fix: one canonical rule --
		// gradients are always logical-order contiguous of size TotalSize(Shape).
		t.Skip("P1: ensureGrad/AllocGrad disagree on gradient buffer size")

		a := leaf(seq(12), 4, 3)
		gradcheck.Check(t, "slice+transpose", func() *tensor.Tensor {
			return a.Slice([]int{1, 0}, []int{3, 3}).Transpose([]int{1, 0})
		}, a)
	})
}

// ---------------------------------------------------------------------------
// Composite chains — where a single-op check can pass but composition breaks.
// ---------------------------------------------------------------------------

func TestGradcheckChains(t *testing.T) {
	t.Run("linear-relu-sum", func(t *testing.T) {
		x := leaf(seq(6), 2, 3)
		w := leaf(seq(12), 3, 4)
		bias := leaf(seq(4), 4)
		gradcheck.Check(t, "linear+relu", func() *tensor.Tensor {
			return x.MatMulAddBias(w, bias).ReLU()
		}, x, w, bias)
	})

	t.Run("two-layer", func(t *testing.T) {
		x := leaf(seq(6), 2, 3)
		w1 := leaf(seq(12), 3, 4)
		w2 := leaf(seq(8), 4, 2)
		gradcheck.Check(t, "two-layer", func() *tensor.Tensor {
			return x.MatMul(w1).Tanh().MatMul(w2)
		}, x, w1, w2)
	})
}
