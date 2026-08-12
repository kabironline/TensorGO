package tensor

import (
	"fmt"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/pools"
	"gonum.org/v1/gonum/mat"
)

// Add performs element-wise addition between two tensors.
func (a *Tensor) Add(b *Tensor) *Tensor {
	// Simple path if shapes match
	if sameShape(a.Shape, b.Shape) {
		outData := a.Device.Allocate(TotalSize(a.Shape))
		a.Device.Add(a.Data(), b.Data(), outData, TotalSize(a.Shape))

		// Create output tensor manually to avoid ToDevice being called on GPU memory
		out := &Tensor{
			data:         StorageFrom(outData),
			Shape:        append([]int{}, a.Shape...),
			Strides:      ComputeStrides(a.Shape),
			Device:       a.Device,
			RequiresGrad: a.RequiresGrad || b.RequiresGrad,
			Parents:      []*Tensor{a, b},
			contiguous:   true,
		}

		out.Backward = func() {
			a.AccumulateGrad(out.Grad())
			b.AccumulateGrad(out.Grad())
		}
		return out
	}

	// General broadcasting path
	out := BroadcastAddOp(a, b)
	out.Parents = []*Tensor{a, b}
	out.Backward = func() {
		// Optimization: Skip ReduceSumTo if shapes match exactly.
		if sameShape(out.Shape, a.Shape) {
			a.AccumulateGrad(out.Grad())
		} else {
			gradA := ReduceSumTo(a.Device, out.Grad(), out.Shape, a.Shape)
			a.AccumulateGrad(gradA)
			a.Device.Free(gradA)
		}

		if sameShape(out.Shape, b.Shape) {
			b.AccumulateGrad(out.Grad())
		} else {
			gradB := ReduceSumTo(b.Device, out.Grad(), out.Shape, b.Shape)
			b.AccumulateGrad(gradB)
			b.Device.Free(gradB)
		}
	}
	return out
}

// sameShape is a helper to compare shapes quickly.
func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Sub performs element-wise subtraction between two tensors.
// It supports broadcasting and uses optimized Gonum paths where possible.
func (a *Tensor) Sub(b *Tensor) *Tensor {
	out := BroadcastSubOp(a, b)
	out.Parents = []*Tensor{a, b}

	out.Backward = func() {
		if sameShape(out.Shape, a.Shape) {
			a.AccumulateGrad(out.Grad())
		} else {
			gradA := ReduceSumTo(a.Device, out.Grad(), out.Shape, a.Shape)
			a.AccumulateGrad(gradA)
			a.Device.Free(gradA)
		}

		// gradB = -1 * out.Grad
		negGrad := a.Device.MulScalar(out.Grad(), -1.0, len(out.Grad()))
		if sameShape(out.Shape, b.Shape) {
			b.AccumulateGrad(negGrad)
			b.Device.Free(negGrad)
		} else {
			gradB := ReduceSumTo(b.Device, negGrad, out.Shape, b.Shape)
			b.AccumulateGrad(gradB)
			b.Device.Free(negGrad)
			b.Device.Free(gradB)
		}
	}
	return out
}

// Mul performs element-wise multiplication between two tensors.
// It supports broadcasting.
func (a *Tensor) Mul(b *Tensor) *Tensor {
	out := BroadcastMulOp(a, b)
	out.Parents = []*Tensor{a, b}

	out.Backward = func() {
		// Grad A = out.Grad * B
		bContig := Contiguous(b)
		tempA := a.Device.BroadcastMul(out.Grad(), bContig.Data(), out.Shape, bContig.Shape, out.Shape)
		if sameShape(out.Shape, a.Shape) {
			a.AccumulateGrad(tempA)
		} else {
			gradA := ReduceSumTo(a.Device, tempA, out.Shape, a.Shape)
			a.AccumulateGrad(gradA)
			a.Device.Free(gradA)
		}
		a.Device.Free(tempA)

		// Grad B = out.Grad * A
		aContig := Contiguous(a)
		tempB := b.Device.BroadcastMul(out.Grad(), aContig.Data(), out.Shape, aContig.Shape, out.Shape)
		if sameShape(out.Shape, b.Shape) {
			b.AccumulateGrad(tempB)
		} else {
			gradB := ReduceSumTo(b.Device, tempB, out.Shape, b.Shape)
			b.AccumulateGrad(gradB)
			b.Device.Free(gradB)
		}
		b.Device.Free(tempB)
		if bContig != b {
			b.Device.Free(bContig.Data())
		}
		if aContig != a {
			a.Device.Free(aContig.Data())
		}
	}
	return out
}

// Div performs element-wise division between two tensors.
// It supports broadcasting.
func (a *Tensor) Div(b *Tensor) *Tensor {
	out := BroadcastDivOp(a, b)
	out.Parents = []*Tensor{a, b}

	out.Backward = func() {
		gradTensor := out.ToGradTensor()

		// Grad A = out.Grad / B
		tempA := BroadcastDivOp(gradTensor, b)
		gradA := ReduceSumTo(a.Device, tempA.Data(), tempA.Shape, a.Shape)
		a.AccumulateGrad(gradA)

		// Grad B = - (out.Grad * A) / (B * B)
		temp := BroadcastMulOp(gradTensor, a)
		temp = BroadcastDivOp(temp, b)
		temp = BroadcastDivOp(temp, b)

		negData := a.Device.MulScalar(temp.Data(), -1.0, len(temp.Data()))
		gradB := ReduceSumTo(b.Device, negData, temp.Shape, b.Shape)
		b.AccumulateGrad(gradB)
	}
	return out
}

// MatMul performs matrix multiplication between two 2D tensors.
func (a *Tensor) MatMul(b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMul only supports 2D matrices")
	}
	if a.Shape[1] != b.Shape[0] {
		panic("Incompatible shapes for matrix multiplication")
	}

	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]

	// creating output tensor
	out := &Tensor{
		data:         StorageFrom(a.Device.Allocate(m * n)),
		Shape:        []int{m, n},
		Strides:      []int{n, 1},
		Device:       a.Device,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		Parents:      []*Tensor{a, b},
		contiguous:   true,
	}

	aOp, releaseA, err := a.asMatOperand()
	if err != nil {
		panic(fmt.Sprintf("MatMul: operand A: %v", err))
	}
	defer releaseA()

	bOp, releaseB, err := b.asMatOperand()
	if err != nil {
		panic(fmt.Sprintf("MatMul: operand B: %v", err))
	}
	defer releaseB()

	a.Device.MatMul(aOp, bOp, out.Data(), 1.0, 0.0)

	// aOp/bOp are released when this function returns, so the closure below must
	// describe a and b again rather than capturing them — a captured operand that
	// took the materializing branch would be a use-after-free by the time
	// BackProp runs.
	out.Backward = func() {
		if out.grad == nil {
			return
		}
		// out is freshly allocated and contiguous, and a gradient is always
		// stored contiguously in logical order.
		gradOut := backend.MatOperand{Data: out.Grad(), Rows: m, Cols: n, LD: n}

		if a.RequiresGrad {
			bOp, release, err := b.asMatOperand()
			if err != nil {
				panic(fmt.Sprintf("MatMul backward: operand B: %v", err))
			}
			defer release()

			// gradA = gradOut @ b^T  ->  (m, k)
			gradA := out.Device.Allocate(m * k)
			out.Device.MatMul(gradOut, bOp.T(), gradA, 1.0, 0.0)
			a.AccumulateGrad(gradA)
			out.Device.Free(gradA)
		}

		if b.RequiresGrad {
			aOp, release, err := a.asMatOperand()
			if err != nil {
				panic(fmt.Sprintf("MatMul backward: operand A: %v", err))
			}
			defer release()

			// gradB = a^T @ gradOut  ->  (k, n)
			gradB := out.Device.Allocate(k * n)
			out.Device.MatMul(aOp.T(), gradOut, gradB, 1.0, 0.0)
			b.AccumulateGrad(gradB)
			out.Device.Free(gradB)
		}
	}

	return out
}

// MatMulAddBias performs matrix multiplication of tensor t with b and adds bias c.
func (t *Tensor) MatMulAddBias(b, c *Tensor) *Tensor {
	if len(t.Shape) != 2 || len(b.Shape) != 2 || len(c.Shape) != 1 {
		panic("MatMulAddBias requires 2D matrices and 1D bias vector")
	}
	if t.Shape[1] != b.Shape[0] || b.Shape[1] != c.Shape[0] {
		panic("Incompatible shapes for MatMulAddBias")
	}

	m, k, n := t.Shape[0], t.Shape[1], b.Shape[1]

	// creating output tensor
	outData := t.Device.Allocate(m * n)

	// out = t @ b
	func() {
		tOp, releaseT, err := t.asMatOperand()
		if err != nil {
			panic(fmt.Sprintf("MatMulAddBias: operand A: %v", err))
		}
		defer releaseT()

		bOp, releaseB, err := b.asMatOperand()
		if err != nil {
			panic(fmt.Sprintf("MatMulAddBias: operand B: %v", err))
		}
		defer releaseB()

		t.Device.MatMul(tOp, bOp, outData, 1.0, 0.0)
	}()

	// Create tensor from matmul result
	matmulOut := &Tensor{
		data:         StorageFrom(outData),
		Shape:        []int{m, n},
		Strides:      []int{n, 1},
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad || b.RequiresGrad,
		Parents:      []*Tensor{t, b},
		contiguous:   true,
	}

	matmulOut.Backward = func() {
		if matmulOut.grad == nil {
			return
		}
		gradOut := backend.MatOperand{Data: matmulOut.Grad(), Rows: m, Cols: n, LD: n}

		if t.RequiresGrad {
			bOp, release, err := b.asMatOperand()
			if err != nil {
				panic(fmt.Sprintf("MatMulAddBias backward: operand B: %v", err))
			}
			defer release()

			// gradT = gradMatMul @ b^T  ->  (m, k)
			gradT := matmulOut.Device.Allocate(m * k)
			matmulOut.Device.MatMul(gradOut, bOp.T(), gradT, 1.0, 0.0)
			t.AccumulateGrad(gradT)
			matmulOut.Device.Free(gradT)
		}

		if b.RequiresGrad {
			tOp, release, err := t.asMatOperand()
			if err != nil {
				panic(fmt.Sprintf("MatMulAddBias backward: operand A: %v", err))
			}
			defer release()

			// gradB = t^T @ gradMatMul  ->  (k, n)
			gradB := matmulOut.Device.Allocate(k * n)
			matmulOut.Device.MatMul(tOp.T(), gradOut, gradB, 1.0, 0.0)
			b.AccumulateGrad(gradB)
			matmulOut.Device.Free(gradB)
		}
	}

	// Add bias using tensor operation (works for both CPU and GPU)
	return matmulOut.Add(c)
}

// MatVecMul performs matrix-vector multiplication.
func (a *Tensor) MatVecMul(b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 1 {
		panic("MatVecMul requires a 2D matrix and a 1D vector")
	}
	if a.Shape[1] != b.Shape[0] {
		panic("Incompatible shapes for matrix-vector multiplication")
	}

	// Treat the vector as an (n, 1) matrix. This must go through Reshape rather
	// than a hand-built &Tensor{}: a literal has no Parents and no Backward, so
	// MatMul's backward would accumulate into a buffer nobody reads and b would
	// silently never train.
	b2D := b.Reshape([]int{b.Shape[0], 1})

	res := a.MatMul(b2D)
	// Flatten result back to 1D
	return res.Reshape([]int{a.Shape[0]})
}

// VecMatMul performs vector-matrix multiplication.
func (a *Tensor) VecMatMul(b *Tensor) *Tensor {
	if len(a.Shape) != 1 || len(b.Shape) != 2 {
		panic("VecMatMul requires a 1D vector and a 2D matrix")
	}
	if a.Shape[0] != b.Shape[0] {
		panic("Incompatible shapes for vector-matrix multiplication")
	}

	// Treat the vector as a (1, m) matrix. See MatVecMul: this must be a Reshape,
	// not a literal, or a's gradient is silently discarded.
	a2D := a.Reshape([]int{1, a.Shape[0]})

	res := a2D.MatMul(b)
	// Flatten result back to 1D
	return res.Reshape([]int{b.Shape[1]})
}

// Dot performs the dot product between two 1D tensors.
func (a *Tensor) Dot(b *Tensor) *Tensor {
	return (a.Mul(b)).Sum()
}

// Inverse computes the inverse of a square 2D tensor (matrix).
func (a *Tensor) Inverse() *Tensor {
	if len(a.Shape) != 2 || a.Shape[0] != a.Shape[1] {
		panic("Inverse requires a square 2D matrix")
	}

	aContig := Contiguous(a)
	aContigF64 := make([]float64, len(aContig.Data()))
	for i, v := range aContig.Data() {
		aContigF64[i] = float64(v)
	}
	mA := mat.NewDense(aContig.Shape[0], aContig.Shape[1], aContigF64)

	var mInv mat.Dense
	err := mInv.Inverse(mA)
	if err != nil {
		panic("Matrix is singular and cannot be inverted")
	}

	resultData := mInv.RawMatrix().Data
	resultDataCopy := pools.GetBuffer(len(resultData))

	for i, v := range resultData {
		resultDataCopy[i] = float32(v)
	}

	out := NewTensor(resultDataCopy, []int{aContig.Shape[0], aContig.Shape[1]}, a)
	// --- AUTOGRAD LOGIC START ---
	out.Backward = func() {
		// Y = X^-1
		// dL/dX = - (Y^T) * GradOut * (Y^T)
		mY := mat.NewDense(out.Shape[0], out.Shape[1], resultData)
		mGradOut := mat.NewDense(out.Shape[0], out.Shape[1], resultData)

		// Temporary: (Y^T) * GradOut
		tmpData := make([]float64, out.Shape[0]*out.Shape[1])
		tmp := mat.NewDense(out.Shape[0], out.Shape[1], tmpData)
		tmp.Mul(mY.T(), mGradOut)

		// Final: tmp * (Y^T)
		finalGradData := make([]float64, out.Shape[0]*out.Shape[1])
		finalGrad := mat.NewDense(out.Shape[0], out.Shape[1], finalGradData)
		finalGrad.Mul(tmp, mY.T())

		a.ensureGrad()
		ag := a.Grad()
		for i, val := range finalGrad.RawMatrix().Data {
			ag[i] -= float32(val) // Note the subtraction (negative sign in formula)
		}
	}
	return out
}

// Slice returns a sub-tensor defined by the provided start and end indexes for each dimension.
func (t *Tensor) Slice(starts, ends []int) *Tensor {
	if len(starts) != len(t.Shape) || len(ends) != len(t.Shape) {
		panic("Slice requires start and end indices for each dimension")
	}

	nd := len(t.Shape)
	// Scalar tensor
	if nd == 0 {
		return t
	}

	// Ensure strides exist (row-major / C-contiguous by default)
	strides := t.Strides
	if strides == nil || len(strides) != nd {
		strides = make([]int, nd)
		strides[nd-1] = 1
		for i := nd - 2; i >= 0; i-- {
			strides[i] = strides[i+1] * t.Shape[i+1]
		}
	}

	offset := 0
	newShape := make([]int, nd)
	fullSlice := true

	for i := 0; i < nd; i++ {
		s := starts[i]
		e := ends[i]

		// support negative indices (Python-style)
		if s < 0 {
			s += t.Shape[i]
		}
		if e < 0 {
			e += t.Shape[i]
		}

		if s < 0 || s > t.Shape[i] || e < 0 || e > t.Shape[i] || s > e {
			panic("invalid slice indices")
		}

		newShape[i] = e - s
		if s != 0 || e != t.Shape[i] {
			fullSlice = false
		}
		offset += s * strides[i]
	}

	// If the slice is the entire tensor, return it directly.
	if fullSlice {
		return t
	}

	out := &Tensor{
		data:         StorageFrom(t.Data()[offset:]), // view starts at offset
		Shape:        newShape,
		Strides:      strides,
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Offset:       t.Offset + offset,
	}
	out.Parents = []*Tensor{t}

	out.Backward = func() {
		// Nothing to propagate if no gradient is present.
		if out.grad == nil || out.grad.Length() == 0 {
			return
		}
		og := out.Grad()

		// Build a full-sized gradient for the parent and place the sliced gradient into it.
		parentGrad := pools.GetBuffer(len(t.Data()))

		// Fast path for scalar-like result (0-d or all dims size 1).
		if len(out.Shape) == 0 || len(og) == 1 && product(out.Shape) == 1 {
			parentGrad[offset] += og[0]
			t.AccumulateGrad(parentGrad)
			pools.PutBuffer(parentGrad)
			return
		}

		// Iterate over coordinates in the output tensor and map them back to parent indices.
		indices := make([]int, len(out.Shape))
		for _, v := range og {
			// compute parent flat index
			pIdx := offset
			for d := range indices {
				pIdx += indices[d] * strides[d]
			}
			parentGrad[pIdx] += v

			// increment multi-dimensional index
			for d := len(indices) - 1; d >= 0; d-- {
				indices[d]++
				if indices[d] < out.Shape[d] {
					break
				}
				indices[d] = 0
			}
		}

		t.AccumulateGrad(parentGrad)
		pools.PutBuffer(parentGrad)
	}

	return out
}

// product returns the product of elements in s (helper used locally).
func product(s []int) int {
	if len(s) == 0 {
		return 1
	}
	prod := 1
	for _, v := range s {
		prod *= v
	}
	return prod
}

// asMatOperand describes t as a 2-D BLAS operand, materializing a contiguous
// copy only when t's strides cannot be expressed as (LD, Trans).
//
// The returned release func is always non-nil, including on the error path, so
// `defer release()` is safe to write unconditionally. It frees the copy if one
// was made and is a no-op otherwise.
func (t *Tensor) asMatOperand() (backend.MatOperand, func(), error) {
	noop := func() {}

	if len(t.Shape) != 2 {
		return backend.MatOperand{}, noop,
			fmt.Errorf("asMatOperand: expected a 2D tensor, got shape %v", t.Shape)
	}

	rows, cols := t.Shape[0], t.Shape[1]
	rowStride, colStride := t.Strides[0], t.Strides[1]

	// Row-major: columns are adjacent, and rows cannot overlap.
	if colStride == 1 && rowStride >= max(1, cols) {
		return backend.MatOperand{
			Data:  t.Data(),
			Rows:  rows,
			Cols:  cols,
			LD:    rowStride,
			Trans: false,
		}, noop, nil
	}

	// Column-major: the stored bytes are a (cols, rows) row-major matrix with
	// leading dimension colStride — exactly what a 2-D transpose produces.
	if rowStride == 1 && colStride >= max(1, rows) {
		return backend.MatOperand{
			Data:  t.Data(),
			Rows:  rows,
			Cols:  cols,
			LD:    colStride,
			Trans: true,
		}, noop, nil
	}

	// Neither layout is addressable by BLAS (e.g. a broadcast view, whose stride
	// is 0 along the broadcast axis): materialize a contiguous copy.
	//
	// Deliberately not via tensor.Contiguous: that returns t unchanged whenever
	// the manually-maintained `contiguous` flag is set, and several view
	// constructors never set it correctly. It also builds a graph node with no
	// Backward, which we have no use for — gradients flow through MatMul's own
	// backward, not through this copy.
	buf := t.Device.Allocate(rows * cols)

	// Offset is passed as 0, not t.Offset: today Slice reslices storage to
	// [offset:] *and* sets Offset, so t.Data() already begins at the logical
	// origin and passing t.Offset again would double-apply it. When P1 fixes
	// Slice to carry Offset without reslicing, this becomes t.Offset and the
	// two t.Data() calls above become t.Data()[t.Offset:].
	t.Device.Contiguous(t.Data(), buf, t.Shape, t.Strides, 0)

	return backend.MatOperand{
		Data:  buf,
		Rows:  rows,
		Cols:  cols,
		LD:    cols,
		Trans: false,
	}, func() { t.Device.Free(buf) }, nil
}
