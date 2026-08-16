// Package gradcheck verifies analytic gradients against finite differences.
//
// The backward pass of an op is hand-written calculus and is easy to get subtly
// wrong — a dropped gradient, a sign flip, a stride misread. The derivative also
// has a definition you can evaluate directly from the forward pass alone:
//
//	df/dx ≈ (f(x+h) − f(x−h)) / 2h
//
// That numeric estimate knows nothing about the chain rule, so it is an
// independent check. Where the two disagree, the backward pass is wrong.
//
// Usage: the inputs are plain contiguous leaf tensors; any view (transpose,
// slice, reshape, broadcast) belongs inside fn, which rebuilds the graph on each
// call. That mirrors how the ops are really used and keeps perturbation simple —
// only leaves are ever written to.
//
//	x := tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
//	x.RequiresGrad = true
//	y := tensor.NewTensor([]float32{7, 8, 9, 10, 11, 12}, []int{2, 3})
//	y.RequiresGrad = true
//
//	gradcheck.Check(t, "matmul/transposed-rhs", func() *tensor.Tensor {
//	    return x.MatMul(y.Transpose([]int{1, 0}))
//	}, x, y)
//
// CPU only: perturbation writes directly into tensor storage, which is not
// addressable from the host for a GPU tensor.
package gradcheck

import (
	"math"
	"testing"

	"github.com/kabironline/nanograd/tensor"
)

// Options tunes the finite-difference comparison.
type Options struct {
	// Eps is the perturbation applied to each input element.
	//
	// Larger than you may expect: these are float32 tensors with ~7 significant
	// digits, and f(x+h) − f(x−h) subtracts two nearly equal numbers. Too small
	// and round-off swamps the signal; too large and the O(h²) truncation error
	// of the central difference dominates. 1e-2 sits in the valley between.
	Eps float64

	// Tol is the maximum acceptable error, measured as
	// |analytic − numeric| / max(1, |analytic|, |numeric|) — relative for large
	// gradients, absolute for small ones, so near-zero entries do not report a
	// spurious 100% disagreement.
	Tol float64

	// MaxReport caps how many mismatching elements are logged per input, so one
	// broken op does not bury the rest of the run. Zero means the default.
	MaxReport int
}

// DefaultOptions are tuned for float32 tensors.
func DefaultOptions() Options {
	return Options{Eps: 1e-2, Tol: 1e-2, MaxReport: 8}
}

// Check verifies the gradients of every input against finite differences,
// reporting a test error for each element that disagrees.
//
// fn must rebuild the graph from the inputs each time it is called and return a
// tensor of any shape; it is reduced with Sum to obtain the scalar loss that the
// gradients are taken with respect to.
func Check(t *testing.T, name string, fn func() *tensor.Tensor, inputs ...*tensor.Tensor) {
	t.Helper()
	CheckWithOptions(t, name, fn, DefaultOptions(), inputs...)
}

// CheckWithOptions is Check with explicit tolerances.
func CheckWithOptions(
	t *testing.T,
	name string,
	fn func() *tensor.Tensor,
	opt Options,
	inputs ...*tensor.Tensor,
) {
	t.Helper()

	// A backward pass that indexes out of range is itself a finding. Contain it
	// so one broken op reports a failure instead of aborting every other check
	// in the run.
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s: panicked during gradient check: %v", name, r)
		}
	}()

	if opt.Eps == 0 {
		opt.Eps = DefaultOptions().Eps
	}
	if opt.Tol == 0 {
		opt.Tol = DefaultOptions().Tol
	}
	if opt.MaxReport == 0 {
		opt.MaxReport = DefaultOptions().MaxReport
	}
	if len(inputs) == 0 {
		t.Fatalf("%s: gradcheck needs at least one input", name)
	}

	for i, in := range inputs {
		if in == nil {
			t.Fatalf("%s: input %d is nil", name, i)
		}
		if !in.RequiresGrad {
			t.Fatalf("%s: input %d does not have RequiresGrad set; "+
				"gradcheck has nothing to verify", name, i)
		}
		if in.Device != nil && in.Device.IsGPU() {
			t.Skipf("%s: gradcheck is CPU-only (input %d is on %s)",
				name, i, in.Device.Name())
		}
	}

	// Reduce with a position-weighted sum rather than a plain Sum.
	//
	// A plain sum is permutation-invariant: d(sum)/dx is 1 for every element no
	// matter what order the op emits them in, so an op that reads a strided view
	// in the wrong order produces an identical loss and gradcheck sees nothing.
	// Weighting each output position by a distinct constant breaks that symmetry.
	weighted := weightedLoss(t, name, fn)

	analytic := analyticGrads(t, name, weighted, inputs)

	for i, in := range inputs {
		data := in.Data()
		reported := 0

		for j := range data {
			numeric := numericGrad(weighted, data, j, opt.Eps)
			got := float64(analytic[i][j])

			if withinTolerance(got, numeric, opt.Tol) {
				continue
			}

			reported++
			if reported > opt.MaxReport {
				t.Errorf("%s: input %d: more mismatches beyond the first %d, "+
					"suppressing the rest", name, i, opt.MaxReport)
				break
			}
			t.Errorf("%s: input %d element %d: analytic=%g numeric=%g "+
				"(rel err %g, tol %g)\n    input shape=%v strides=%v",
				name, i, j, got, numeric, relError(got, numeric), opt.Tol,
				in.Shape, in.Strides)
		}
	}
}

// weightedLoss wraps fn so the graph ends in sum(out * w), where w is a fixed
// tensor of distinct constants that does not require gradients.
//
// w is built once from the shape of an initial fn() call and captured, so every
// perturbed forward pass sees the identical weights.
func weightedLoss(t *testing.T, name string, fn func() *tensor.Tensor) func() *tensor.Tensor {
	t.Helper()

	probe := fn()
	if probe == nil {
		t.Fatalf("%s: fn returned nil", name)
	}

	n := tensor.TotalSize(probe.Shape)
	wData := make([]float32, n)
	for i := range wData {
		// Distinct, non-zero, and not monotonically related to the inputs, so a
		// transposed read produces a visibly different weighted sum.
		wData[i] = float32(i%7)*0.41 + 1.13
	}
	w := tensor.NewTensor(wData, append([]int(nil), probe.Shape...))
	w.RequiresGrad = false

	return func() *tensor.Tensor { return fn().Mul(w) }
}

// analyticGrads runs one forward/backward pass and snapshots each input's
// gradient. The snapshot matters: later calls to fn rebuild the graph, and a
// live gradient slice could be reallocated underneath us.
func analyticGrads(
	t *testing.T,
	name string,
	fn func() *tensor.Tensor,
	inputs []*tensor.Tensor,
) [][]float32 {
	t.Helper()

	for _, in := range inputs {
		in.ZeroGrad()
	}

	out := fn()
	if out == nil {
		t.Fatalf("%s: fn returned nil", name)
	}
	out.Sum().BackProp()

	grads := make([][]float32, len(inputs))
	for i, in := range inputs {
		g := in.Grad()
		if g == nil {
			t.Fatalf("%s: input %d has no gradient after BackProp — "+
				"the backward pass dropped it entirely", name, i)
		}

		n := len(in.Data())
		if len(g) < n {
			t.Fatalf("%s: input %d gradient has %d elements but data has %d",
				name, i, len(g), n)
		}

		grads[i] = append([]float32(nil), g[:n]...)
	}
	return grads
}

// numericGrad estimates d(sum(fn))/d(data[j]) by central difference, restoring
// data[j] before returning.
func numericGrad(fn func() *tensor.Tensor, data []float32, j int, eps float64) float64 {
	orig := data[j]

	data[j] = orig + float32(eps)
	plus := lossOf(fn)

	data[j] = orig - float32(eps)
	minus := lossOf(fn)

	data[j] = orig
	return (plus - minus) / (2 * eps)
}

// lossOf rebuilds the graph and reduces it to the same scalar the analytic pass
// differentiates — via Sum, not by ranging over out.Data(), because for a
// non-contiguous output the raw storage is not the logical values.
//
// The result is widened to float64 immediately: the two perturbed losses differ
// only in the low bits, and doing the subtraction in float32 would lose most of
// the signal to cancellation.
func lossOf(fn func() *tensor.Tensor) float64 {
	return float64(fn().Sum().Data()[0])
}

func relError(a, b float64) float64 {
	den := math.Max(1.0, math.Max(math.Abs(a), math.Abs(b)))
	return math.Abs(a-b) / den
}

func withinTolerance(a, b, tol float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}
	return relError(a, b) <= tol
}
