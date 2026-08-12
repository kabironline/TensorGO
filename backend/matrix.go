package backend

// MatOperand describes a 2-D matrix as BLAS sees it.
//
// Rows and Cols are the *logical* dimensions — the shape this operand has in
// the multiplication. Trans describes how the bytes are laid out: when true the
// stored matrix is the transpose of the logical one, i.e. (Cols, Rows) in
// row-major order with leading dimension LD. Backends swap the dimensions
// themselves when building their BLAS descriptor.
//
// One representation covers two things that used to be separate: a transposed
// *view* (Tensor.Transpose, which moves no bytes) and a mathematical transpose
// requested by a caller (T below). They are the same operation, so they compose
// — and cancel — for free.
type MatOperand struct {
	Data  []float32 // becomes *storage.Storage in P3
	Rows  int       // logical rows
	Cols  int       // logical cols
	LD    int       // leading dimension of the stored matrix
	Trans bool      // stored layout is the transpose of the logical one
}

// T returns an operand over the same bytes whose logical value is the
// transpose. Nothing is copied and no memory is touched.
func (op MatOperand) T() MatOperand {
	op.Rows, op.Cols = op.Cols, op.Rows
	op.Trans = !op.Trans
	return op
}

// StoredDims returns the dimensions of the matrix as it is actually laid out in
// memory, which is what a BLAS descriptor needs.
func (op MatOperand) StoredDims() (rows, cols int) {
	if op.Trans {
		return op.Cols, op.Rows
	}
	return op.Rows, op.Cols
}

type MatrixOps interface {
	// MatMul computes out = alpha*(A @ B) + beta*out.
	//
	// A is a.Rows x a.Cols and B is b.Rows x b.Cols logically; a.Cols must equal
	// b.Rows. out is contiguous, a.Rows x b.Cols, with leading dimension b.Cols.
	//
	// There is deliberately no MatMulTransA/MatMulTransB/MatMulAdd: transposition
	// belongs to the operand (MatOperand.T) and accumulation to beta. Separate
	// methods made the operand's stored layout and the operation's mathematical
	// transpose two different flags that silently disagreed.
	MatMul(a, b MatOperand, out []float32, alpha, beta float32)
}
