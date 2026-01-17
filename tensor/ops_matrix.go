package tensor

import (
	"github.com/kabironline/nanograd/internal/pools"
	"gonum.org/v1/gonum/mat"
)

// Add performs element-wise addition between two tensors.
func (a *Tensor) Add(b *Tensor) *Tensor {
	// Simple path if shapes match
	if sameShape(a.Shape, b.Shape) {
		outData := a.Device.Allocate(len(a.Data))
		a.Device.Add(a.Data, b.Data, outData, len(a.Data))
		out := NewTensor(outData, a.Shape, a, b)
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
			a.AccumulateGrad(ReduceSumTo(a.Device, out.Grad, out.Shape, a.Shape))
		}

		if sameShape(out.Shape, b.Shape) {
			b.AccumulateGrad(out.Grad)
		} else {
			b.AccumulateGrad(ReduceSumTo(b.Device, out.Grad, out.Shape, b.Shape))
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
		gradA := ReduceSumTo(a.Device, out.Grad, out.Shape, a.Shape)
		a.AccumulateGrad(gradA)

		// gradB = -1 * out.Grad
		negGrad := a.Device.MulScalar(out.Grad, -1.0, len(out.Grad))
		gradB := ReduceSumTo(b.Device, negGrad, out.Shape, b.Shape)
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
		gradTensor := out.ToGradTensor()

		// Grad A = out.Grad * B
		tempA := BroadcastMulOp(gradTensor, b)
		gradA := ReduceSumTo(a.Device, tempA.Data, tempA.Shape, a.Shape)
		a.AccumulateGrad(gradA)

		// Grad B = out.Grad * A
		tempB := BroadcastMulOp(gradTensor, a)
		gradB := ReduceSumTo(b.Device, tempB.Data, tempB.Shape, b.Shape)
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
		gradTensor := out.ToGradTensor()

		// Grad A = out.Grad / B
		tempA := BroadcastDivOp(gradTensor, b)
		gradA := ReduceSumTo(a.Device, tempA.Data, tempA.Shape, a.Shape)
		a.AccumulateGrad(gradA)

		// Grad B = - (out.Grad * A) / (B * B)
		temp := BroadcastMulOp(gradTensor, a)
		temp = BroadcastDivOp(temp, b)
		temp = BroadcastDivOp(temp, b)

		negData := a.Device.MulScalar(temp.Data, -1.0, len(temp.Data))
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
		Data:         a.Device.Allocate(m * n),
		Shape:        []int{m, n},
		Strides:      []int{n, 1},
		Device:       a.Device,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}

	a.Device.MatMul(a.Data, b.Data, out.Data, m, n, k, a.Strides[0], b.Strides[0])

	out.Backward = func() {
		// gradA = gradOut @ b^T
		gradA := out.Device.Allocate(len(a.Data))
		out.Device.MatMulTransB(out.Grad, b.Data, gradA, m, k, n, out.Strides[0], b.Strides[0])
		a.AccumulateGrad(gradA)

		// gradB = a^T @ gradOut
		gradB := out.Device.Allocate(len(b.Data))
		out.Device.MatMulTransA(a.Data, out.Grad, gradB, k, n, m, a.Strides[0], out.Strides[0])
		b.AccumulateGrad(gradB)
	}

	return out
}

// MatMulAddBias performs matrix multiplication of tensor t with b and adds bias c.
func (t *Tensor) MatMulAddBias(b, c *Tensor) *Tensor {
	return t.MatMul(b).Add(c)
}

// MatVecMul performs matrix-vector multiplication.
func (a *Tensor) MatVecMul(b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 1 {
		panic("MatVecMul requires a 2D matrix and a 1D vector")
	}
	if a.Shape[1] != b.Shape[0] {
		panic("Incompatible shapes for matrix-vector multiplication")
	}

	// Treat vector as (n, 1) matrix
	b2D := &Tensor{
		Data:         b.Data,
		Shape:        []int{b.Shape[0], 1},
		Strides:      []int{1, 1},
		Device:       b.Device,
		RequiresGrad: b.RequiresGrad,
	}
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

	// Treat vector as (1, m) matrix
	a2D := &Tensor{
		Data:         a.Data,
		Shape:        []int{1, a.Shape[0]},
		Strides:      []int{a.Shape[0], 1},
		Device:       a.Device,
		RequiresGrad: a.RequiresGrad,
	}
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
	aContigF64 := make([]float64, len(aContig.Data))
	for i, v := range aContig.Data {
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

		for i, val := range finalGrad.RawMatrix().Data {
			a.Grad[i] -= float32(val) // Note the subtraction (negative sign in formula)
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
		Data:         t.Data[offset:], // view starts at offset
		Shape:        newShape,
		Strides:      strides,
		Device:       t.Device,
		RequiresGrad: t.RequiresGrad,
		Offset:       t.Offset + offset,
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
