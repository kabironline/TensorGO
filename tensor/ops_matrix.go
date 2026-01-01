package tensor

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Add performs element-wise addition between two tensors.
// Optimized with a fast-path for identical shapes using Gonum SIMD and minimizing reduction overhead.
func (a *Tensor) Add(b *Tensor) *Tensor {
	// Fast path: Identical shapes avoid broadcasting logic and use SIMD.
	if sameShape(a.Shape, b.Shape) {
		res := make([]float64, len(a.Data))
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
		negGrad := make([]float64, len(out.Grad))
		for i, v := range out.Grad {
			negGrad[i] = -v
		}
		gradB := ReduceSumTo(negGrad, out.Shape, b.Shape)
		b.AccumulateGrad(gradB)
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

		negData := make([]float64, len(temp.Data))
		for i, v := range temp.Data {
			negData[i] = -v
		}

		gradB := ReduceSumTo(negData, temp.Shape, b.Shape)
		b.AccumulateGrad(gradB)
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
	resultData := make([]float64, m*p)
	mC := mat.NewDense(m, p, resultData)

	// Optimized BLAS call
	mC.Mul(mA, mB)

	out := NewTensor(resultData, []int{m, p}, a, b)

	// --- AUTOGRAD LOGIC START ---
	out.Backward = func() {
		// Wrap gradients and data in Gonum Dense views.
		// .T() is a zero-copy metadata operation in Gonum.
		mGradOut := mat.NewDense(out.Shape[0], out.Shape[1], out.Grad)
		mAC := mat.NewDense(aContig.Shape[0], aContig.Shape[1], aContig.Data)
		mBC := mat.NewDense(bContig.Shape[0], bContig.Shape[1], bContig.Data)

		// 1. Grad A = GradOut * B^T
		// Avoids the expensive Transpose() + Contiguous() copy cycle.
		var mGradA mat.Dense
		mGradA.Mul(mGradOut, mBC.T())
		a.AccumulateGrad(mGradA.RawMatrix().Data)

		// 2. Grad B = A^T * GradOut
		var mGradB mat.Dense
		mGradB.Mul(mAC.T(), mGradOut)
		b.AccumulateGrad(mGradB.RawMatrix().Data)
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

	resultData := make([]float64, aContig.Shape[0])
	vC := mat.NewVecDense(aContig.Shape[0], resultData)
	vC.MulVec(mA, vB)

	out := NewTensor(resultData, []int{aContig.Shape[0]}, a, b)

	// --- AUTOGRAD LOGIC START ---
	out.Backward = func() {
		// Wrap for Gonum
		mGradOut := mat.NewVecDense(out.Shape[0], out.Grad)
		vB := mat.NewVecDense(b.Shape[0], b.Data)
		mA := mat.NewDense(a.Shape[0], a.Shape[1], a.Data)

		// 1. Grad A = GradOut (m x 1) * b^T (1 x n) -> (m x n)
		// This is an outer product
		mGradA := mat.NewDense(a.Shape[0], a.Shape[1], make([]float64, len(a.Data)))
		mGradA.Outer(1, mGradOut, vB)
		for i, val := range mGradA.RawMatrix().Data {
			a.Grad[i] += val
		}

		// 2. Grad B = a^T * GradOut
		vGradB := mat.NewVecDense(b.Shape[0], make([]float64, len(b.Data)))
		vGradB.MulVec(mA.T(), mGradOut)
		for i, val := range vGradB.RawVector().Data {
			b.Grad[i] += val
		}
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

	resultData := make([]float64, bContig.Shape[1])
	vC := mat.NewVecDense(bContig.Shape[1], resultData)
	vC.MulVec(mB.T(), vA)

	out := NewTensor(resultData, []int{bContig.Shape[1]}, a, b)

	// --- AUTOGRAD LOGIC START ---

	out.Backward = func() {
		vGradOut := mat.NewVecDense(out.Shape[0], out.Grad)
		vA := mat.NewVecDense(a.Shape[0], a.Data)
		mB := mat.NewDense(b.Shape[0], b.Shape[1], b.Data)

		// 1. Grad A = GradOut * B^T
		vGradA := mat.NewVecDense(a.Shape[0], make([]float64, len(a.Data)))
		vGradA.MulVec(mB, vGradOut) // MulVec(m, v) is m * v
		for i, val := range vGradA.RawVector().Data {
			a.Grad[i] += val
		}

		// 2. Grad B = a^T (outer) GradOut
		mGradB := mat.NewDense(b.Shape[0], b.Shape[1], make([]float64, len(b.Data)))
		mGradB.Outer(1, vA, vGradOut)
		for i, val := range mGradB.RawMatrix().Data {
			b.Grad[i] += val
		}
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
		var tmp mat.Dense
		tmp.Mul(mY.T(), mGradOut)

		// Final: tmp * (Y^T)
		var finalGrad mat.Dense
		finalGrad.Mul(&tmp, mY.T())

		for i, val := range finalGrad.RawMatrix().Data {
			a.Grad[i] -= val // Note the subtraction (negative sign in formula)
		}
	}
	return out
}
