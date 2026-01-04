package tensor

import (
	"github.com/kabironline/nanograd/internal/pools"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Add performs element-wise addition between two tensors.
// Optimized with a fast-path for identical shapes using Gonum SIMD and minimizing reduction overhead.
func (a *Tensor) Add(b *Tensor) *Tensor {
	// Fast path: Identical shapes avoid broadcasting logic and use SIMD.
	if sameShape(a.Shape, b.Shape) {
		res := pools.GetBuffer(len(a.Data))
		floats.AddTo(res, a.Data, b.Data)
		out := NewTensor(res, a.Shape, a, b)
		out.Backward = func() {
			a.AccumulateGrad(out.Grad)
			b.AccumulateGrad(out.Grad)
		}
		return out
	}

	// General broadcasting path
	out := BroadcastAddOp(a, b)
	out.Parents = []*Tensor{a, b}
	out.Backward = func() {
		// Optimization: Skip ReduceSumTo if shapes match exactly.
		if sameShape(out.Shape, a.Shape) {
			a.AccumulateGrad(out.Grad)
		} else {
			a.AccumulateGrad(ReduceSumTo(out.Grad, out.Shape, a.Shape))
		}

		if sameShape(out.Shape, b.Shape) {
			b.AccumulateGrad(out.Grad)
		} else {
			b.AccumulateGrad(ReduceSumTo(out.Grad, out.Shape, b.Shape))
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
		gradA := ReduceSumTo(out.Grad, out.Shape, a.Shape)
		a.AccumulateGrad(gradA)

		// gradB = -1 * out.Grad
		negGrad := pools.GetBuffer(len(out.Grad))
		for i, v := range out.Grad {
			negGrad[i] = -v
		}
		gradB := ReduceSumTo(negGrad, out.Shape, b.Shape)
		b.AccumulateGrad(gradB)
		pools.PutBuffer(negGrad)
	}
	return out
}

// Mul performs element-wise multiplication between two tensors.
// It supports broadcasting.
func (a *Tensor) Mul(b *Tensor) *Tensor {
	out := BroadcastMulOp(a, b)
	out.Parents = []*Tensor{a, b}

	out.Backward = func() {
		gradTensor := &Tensor{Data: out.Grad, Shape: out.Shape, Strides: out.Strides}

		// Grad A = out.Grad * B
		tempA := BroadcastMulOp(gradTensor, b)
		gradA := ReduceSumTo(tempA.Data, tempA.Shape, a.Shape)
		a.AccumulateGrad(gradA)

		// Grad B = out.Grad * A
		tempB := BroadcastMulOp(gradTensor, a)
		gradB := ReduceSumTo(tempB.Data, tempB.Shape, b.Shape)
		b.AccumulateGrad(gradB)
	}
	return out
}

// Div performs element-wise division between two tensors.
// It supports broadcasting.
func (a *Tensor) Div(b *Tensor) *Tensor {
	out := BroadcastDivOp(a, b)
	out.Parents = []*Tensor{a, b}

	out.Backward = func() {
		gradTensor := &Tensor{Data: out.Grad, Shape: out.Shape, Strides: out.Strides}

		// Grad A = out.Grad / B
		tempA := BroadcastDivOp(gradTensor, b)
		gradA := ReduceSumTo(tempA.Data, tempA.Shape, a.Shape)
		a.AccumulateGrad(gradA)

		// Grad B = - (out.Grad * A) / (B * B)
		temp := BroadcastMulOp(gradTensor, a)
		temp = BroadcastDivOp(temp, b)
		temp = BroadcastDivOp(temp, b)

		negData := pools.GetBuffer(len(temp.Data))
		for i, v := range temp.Data {
			negData[i] = -v
		}

		gradB := ReduceSumTo(negData, temp.Shape, b.Shape)
		b.AccumulateGrad(gradB)
		pools.PutBuffer(negData)
	}
	return out
}

// MatMul performs matrix multiplication between two 2D tensors.
// It ensures both tensors are contiguous before calling Gonum's BLAS-optimized Mul.
func (a *Tensor) MatMul(b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMul only supports 2D matrices")
	}
	if a.Shape[1] != b.Shape[0] {
		panic("Incompatible shapes for matrix multiplication")
	}

	// Ensure inputs are contiguous for Gonum; if already contiguous, this is a no-op.
	aContig := Contiguous(a)
	bContig := Contiguous(b)

	mA := mat.NewDense(aContig.Shape[0], aContig.Shape[1], aContig.Data)
	mB := mat.NewDense(bContig.Shape[0], bContig.Shape[1], bContig.Data)

	m, p := aContig.Shape[0], bContig.Shape[1]
	resultData := pools.GetBuffer(m * p)
	mC := mat.NewDense(m, p, resultData)

	// Optimized BLAS call
	mC.Mul(mA, mB)

	out := NewTensor(resultData, []int{m, p}, a, b)
	pools.PutBuffer(resultData)

	// --- AUTOGRAD LOGIC START ---
	out.Backward = func() {
		// Wrap gradients and data in Gonum Dense views.
		// .T() is a zero-copy metadata operation in Gonum.
		mGradOut := mat.NewDense(out.Shape[0], out.Shape[1], out.Grad)
		mAC := mat.NewDense(aContig.Shape[0], aContig.Shape[1], aContig.Data)
		mBC := mat.NewDense(bContig.Shape[0], bContig.Shape[1], bContig.Data)

		// 1. Grad A = GradOut * B^T
		// Avoids the expensive Transpose() + Contiguous() copy cycle.
		gradAData := pools.GetBuffer(aContig.Shape[0] * aContig.Shape[1])
		mGradA := mat.NewDense(aContig.Shape[0], aContig.Shape[1], gradAData)
		mGradA.Mul(mGradOut, mBC.T())
		a.AccumulateGrad(mGradA.RawMatrix().Data)
		pools.PutBuffer(gradAData)

		// 2. Grad B = A^T * GradOut
		gradBData := pools.GetBuffer(bContig.Shape[0] * bContig.Shape[1])
		mGradB := mat.NewDense(bContig.Shape[0], bContig.Shape[1], gradBData)
		mGradB.Mul(mAC.T(), mGradOut)
		b.AccumulateGrad(mGradB.RawMatrix().Data)
		pools.PutBuffer(gradBData)
	}
	// --- AUTOGRAD LOGIC END ---

	return out
}

// MatMulAddBias performs matrix multiplication of tensor t with b and adds bias c.
// It leverages optimized Gonum paths and supports autograd.
func (t *Tensor) MatMulAddBias(b, c *Tensor) *Tensor {
	// t: (m, n), b: (n, p), c: (p,) or (1, p) or (p)
	if len(t.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMulAddBias: t and b must be 2D tensors")
	}
	if t.Shape[1] != b.Shape[0] {
		panic("MatMulAddBias: shapes of t and b not compatible for matmul")
	}
	m, n := t.Shape[0], t.Shape[1]
	p := b.Shape[1]

	// Ensure contiguity for Gonum
	tContig := Contiguous(t)
	bContig := Contiguous(b)

	mT := mat.NewDense(tContig.Shape[0], tContig.Shape[1], tContig.Data)
	mB := mat.NewDense(bContig.Shape[0], bContig.Shape[1], bContig.Data)

	resultData := pools.GetBuffer(m * p)
	mOut := mat.NewDense(m, p, resultData)
	mOut.Mul(mT, mB)

	// Add bias c (broadcast over rows)
	// c can be shape (p,), (1, p), or (p)
	var bias []float64
	if len(c.Shape) == 1 && c.Shape[0] == p {
		bias = c.Data
	} else if len(c.Shape) == 2 && c.Shape[0] == 1 && c.Shape[1] == p {
		bias = c.Data
	} else if len(c.Shape) == 0 && p == 1 {
		bias = c.Data
	} else {
		panic("MatMulAddBias: bias shape not compatible")
	}
	for i := range m {
		for j := range p {
			mOut.Set(i, j, mOut.At(i, j)+bias[j])
		}
	}

	out := NewTensor(resultData, []int{m, p}, t, b, c)
	pools.PutBuffer(resultData)

	// --- AUTOGRAD LOGIC START ---
	out.Backward = func() {
		// out.Grad: shape (m, p)
		mGradOut := mat.NewDense(m, p, out.Grad)
		mT := mat.NewDense(tContig.Shape[0], tContig.Shape[1], tContig.Data)
		mB := mat.NewDense(bContig.Shape[0], bContig.Shape[1], bContig.Data)

		// 1. Grad t = GradOut * b^T
		gradTData := pools.GetBuffer(m * n)
		mGradT := mat.NewDense(m, n, gradTData)
		mGradT.Mul(mGradOut, mB.T())
		t.AccumulateGrad(mGradT.RawMatrix().Data)
		pools.PutBuffer(gradTData)

		// 2. Grad b = t^T * GradOut
		gradBData := pools.GetBuffer(n * p)
		mGradB := mat.NewDense(n, p, gradBData)
		mGradB.Mul(mT.T(), mGradOut)
		b.AccumulateGrad(mGradB.RawMatrix().Data)
		pools.PutBuffer(gradBData)

		// 3. Grad c = sum over rows of GradOut (broadcasted add)
		gradC := pools.GetBuffer(p)
		for i := range mGradOut.RawMatrix().Data {
			row := i / p
			col := i % p
			if row < m && col < p {
				gradC[col] += mGradOut.RawMatrix().Data[i]
			}
		}
		// Accumulate into c.Grad, handling broadcast shape
		if len(c.Shape) == 1 && c.Shape[0] == p {
			for j := range gradC {
				c.Grad[j] += gradC[j]
			}
		} else if len(c.Shape) == 2 && c.Shape[0] == 1 && c.Shape[1] == p {
			for j := range gradC {
				c.Grad[j] += gradC[j]
			}
		} else if len(c.Shape) == 0 && p == 1 {
			c.Grad[0] += gradC[0]
		} else {
			panic("MatMulAddBias: bias shape not compatible in backward")
		}
		pools.PutBuffer(gradC)
	}
	// --- AUTOGRAD LOGIC END ---

	return out
}

// MatVecMul performs matrix-vector multiplication.
// It ensures contiguity to use optimized Gonum paths.
func (a *Tensor) MatVecMul(b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 1 {
		panic("MatVecMul requires a 2D matrix and a 1D vector")
	}
	if a.Shape[1] != b.Shape[0] {
		panic("Incompatible shapes for matrix-vector multiplication")
	}

	aContig := Contiguous(a)
	bContig := Contiguous(b)

	mA := mat.NewDense(aContig.Shape[0], aContig.Shape[1], aContig.Data)
	vB := mat.NewVecDense(bContig.Shape[0], bContig.Data)

	resultData := pools.GetBuffer(aContig.Shape[0])
	vC := mat.NewVecDense(aContig.Shape[0], resultData)
	vC.MulVec(mA, vB)

	out := NewTensor(resultData, []int{aContig.Shape[0]}, a, b)
	pools.PutBuffer(resultData)

	// --- AUTOGRAD LOGIC START ---
	out.Backward = func() {
		// Wrap for Gonum
		mGradOut := mat.NewVecDense(out.Shape[0], out.Grad)
		vB := mat.NewVecDense(b.Shape[0], b.Data)
		mA := mat.NewDense(a.Shape[0], a.Shape[1], a.Data)

		// 1. Grad A = GradOut (m x 1) * b^T (1 x n) -> (m x n)
		// This is an outer product
		gradAData := pools.GetBuffer(len(a.Data))
		mGradA := mat.NewDense(a.Shape[0], a.Shape[1], gradAData)
		mGradA.Outer(1, mGradOut, vB)
		for i, val := range mGradA.RawMatrix().Data {
			a.Grad[i] += val
		}
		pools.PutBuffer(gradAData)

		// 2. Grad B = a^T * GradOut
		gradBData := pools.GetBuffer(len(b.Data))
		vGradB := mat.NewVecDense(b.Shape[0], gradBData)
		vGradB.MulVec(mA.T(), mGradOut)
		for i, val := range vGradB.RawVector().Data {
			b.Grad[i] += val
		}
		pools.PutBuffer(gradBData)
	}
	return out
}

// VecMatMul performs vector-matrix multiplication.
func (a *Tensor) VecMatMul(b *Tensor) *Tensor {
	if len(a.Shape) != 1 || len(b.Shape) != 2 {
		panic("VecMatMul requires a 1D vector and a 2D matrix")
	}
	if a.Shape[0] != b.Shape[0] {
		panic("Incompatible shapes for vector-matrix multiplication")
	}

	aContig := Contiguous(a)
	bContig := Contiguous(b)

	vA := mat.NewVecDense(aContig.Shape[0], aContig.Data)
	mB := mat.NewDense(bContig.Shape[0], bContig.Shape[1], bContig.Data)

	resultData := pools.GetBuffer(bContig.Shape[1])
	vC := mat.NewVecDense(bContig.Shape[1], resultData)
	vC.MulVec(mB.T(), vA)

	out := NewTensor(resultData, []int{bContig.Shape[1]}, a, b)
	pools.PutBuffer(resultData)
	// --- AUTOGRAD LOGIC START ---

	out.Backward = func() {
		vGradOut := mat.NewVecDense(out.Shape[0], out.Grad)
		vA := mat.NewVecDense(a.Shape[0], a.Data)
		mB := mat.NewDense(b.Shape[0], b.Shape[1], b.Data)

		// 1. Grad A = GradOut * B^T
		gradAData := pools.GetBuffer(len(a.Data))
		vGradA := mat.NewVecDense(a.Shape[0], gradAData)
		vGradA.MulVec(mB, vGradOut) // MulVec(m, v) is m * v
		for i, val := range vGradA.RawVector().Data {
			a.Grad[i] += val
		}
		pools.PutBuffer(gradAData)

		// 2. Grad B = a^T (outer) GradOut
		gradBData := pools.GetBuffer(len(b.Data))
		mGradB := mat.NewDense(b.Shape[0], b.Shape[1], gradBData)
		mGradB.Outer(1, vA, vGradOut)
		for i, val := range mGradB.RawMatrix().Data {
			b.Grad[i] += val
		}
		pools.PutBuffer(gradBData)
	}
	return out
}

// Dot performs the dot product between two 1D tensors.
func (a *Tensor) Dot(b *Tensor) *Tensor {
	if len(a.Shape) != 1 || len(b.Shape) != 1 {
		panic("Dot requires two 1D vectors")
	}
	if a.Shape[0] != b.Shape[0] {
		panic("Incompatible shapes for dot product")
	}

	aContig := Contiguous(a)
	bContig := Contiguous(b)

	vA := mat.NewVecDense(aContig.Shape[0], aContig.Data)
	vB := mat.NewVecDense(bContig.Shape[0], bContig.Data)
	result := mat.Dot(vA, vB)

	out := NewTensor([]float64{result}, []int{}, a, b)
	// --- AUTOGRAD LOGIC START ---
	out.Backward = func() {
		// Grad of Dot product: gradA = gradOut * b, gradB = gradOut * a
		gradOutScalar := out.Grad[0]
		for i := range a.Grad {
			a.Grad[i] += gradOutScalar * b.Data[i]
			b.Grad[i] += gradOutScalar * a.Data[i]
		}
	}
	return out
}

// Inverse computes the inverse of a square 2D tensor (matrix).
func (a *Tensor) Inverse() *Tensor {
	if len(a.Shape) != 2 || a.Shape[0] != a.Shape[1] {
		panic("Inverse requires a square 2D matrix")
	}

	aContig := Contiguous(a)
	mA := mat.NewDense(aContig.Shape[0], aContig.Shape[1], aContig.Data)

	var mInv mat.Dense
	err := mInv.Inverse(mA)
	if err != nil {
		panic("Matrix is singular and cannot be inverted")
	}

	resultData := mInv.RawMatrix().Data
	out := NewTensor(resultData, []int{aContig.Shape[0], aContig.Shape[1]}, a)
	// --- AUTOGRAD LOGIC START ---
	out.Backward = func() {
		// Y = X^-1
		// dL/dX = - (Y^T) * GradOut * (Y^T)
		mY := mat.NewDense(out.Shape[0], out.Shape[1], out.Data)
		mGradOut := mat.NewDense(out.Shape[0], out.Shape[1], out.Grad)

		// Temporary: (Y^T) * GradOut
		tmpData := pools.GetBuffer(out.Shape[0] * out.Shape[1])
		tmp := mat.NewDense(out.Shape[0], out.Shape[1], tmpData)
		tmp.Mul(mY.T(), mGradOut)

		// Final: tmp * (Y^T)
		finalGradData := pools.GetBuffer(out.Shape[0] * out.Shape[1])
		finalGrad := mat.NewDense(out.Shape[0], out.Shape[1], finalGradData)
		finalGrad.Mul(tmp, mY.T())

		for i, val := range finalGrad.RawMatrix().Data {
			a.Grad[i] -= val // Note the subtraction (negative sign in formula)
		}
		pools.PutBuffer(tmpData)
		pools.PutBuffer(finalGradData)
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
		Data:    t.Data[offset:], // view starts at offset
		Shape:   newShape,
		Strides: strides,
	}
	out.Parents = []*Tensor{t}

	out.Backward = func() {
		// Nothing to propagate if no gradient is present.
		if len(out.Grad) == 0 {
			return
		}

		// Build a full-sized gradient for the parent and place the sliced gradient into it.
		parentGrad := pools.GetBuffer(len(t.Data))

		// Fast path for scalar-like result (0-d or all dims size 1).
		if len(out.Shape) == 0 || len(out.Grad) == 1 && product(out.Shape) == 1 {
			parentGrad[offset] += out.Grad[0]
			t.AccumulateGrad(parentGrad)
			pools.PutBuffer(parentGrad)
			return
		}

		// Iterate over coordinates in the output tensor and map them back to parent indices.
		indices := make([]int, len(out.Shape))
		for _, v := range out.Grad {
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
