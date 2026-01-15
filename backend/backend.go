package backend

// ============================================================================
// Core Backend Interface - Minimal Set of Required Operations
// ============================================================================

// Backend defines the minimal interface that all computation backends must implement.
// Use composition and optional interfaces to avoid forcing backends to implement everything.
type Backend interface {
	// Name returns the backend identifier (e.g., "cpu", "cuda:0", "rocm:1")
	Name() string

	// IsGPU returns true if this is a GPU backend
	IsGPU() bool

	// Memory management
	MemoryManager

	// Core operations
	ElementWiseOps
	MatrixOps
	ReductionOps
	AxisReductionOps
	ConvOps
	PoolOps

	// Common neural network operations
	ActivationOps
	SoftmaxOps
	BroadcastOps
	UtilityOps
	RandomOps
	OptimizerOps
}

type OptimizerOps interface {
	// StepSGD performs a single step of SGD update: data -= lr * grad
	StepSGD(data, grad []float64, lr float64)

	// StepAdam performs a single step of Adam update
	StepAdam(data, grad, m, v []float64, lr, beta1, beta2, eps float64, t int)
}

type RandomOps interface {
	// Normal fills the buffer with normal distribution values
	Normal(data []float64, mean, stdDev float64, size int)
}

type ConvOps interface {
	// Im2Col transforms an image tensor to a column matrix
	// data: [N, C, H, W]
	// returns: [C*kH*kW, N*outH*outW]
	Im2Col(data []float64, shape, strides []int, kH, kW, stride, padding int) []float64

	// Col2Im transforms a column matrix back to an image tensor (gradient)
	// colGrad: [C*kH*kW, N*outH*outW]
	// returns: [N, C, H, W]
	Col2Im(colGrad []float64, xShape, xStrides []int, kH, kW, stride, padding int) []float64
}

// ============================================================================
// Pooling Operations Interface
// ============================================================================

type PoolOps interface {
	// MaxPool2d performs 2D max pooling
	// data: [N, C, H, W]
	// returns: [N, C, outH, outW] and indices for backward pass
	MaxPool2d(data []float64, shape, strides []int, kH, kW, stride, padding int) ([]float64, []int)

	// MaxPool2dBackward computes gradient for 2D max pooling
	MaxPool2dBackward(grad []float64, indices []int, xShape []int) []float64
}

// ============================================================================
// Memory Management Interface
// ============================================================================

type MemoryManager interface {
	// Allocate creates a new buffer of the given size
	// Returns a slice that may point to GPU memory (for GPU backends)
	Allocate(size int) []float64

	// Free releases memory (no-op for CPU, important for GPU)
	Free(data []float64)

	// Copy performs an optimized copy (same device)
	Copy(dst, src []float64)
}

// MemoryTransfer is an optional interface for backends that support device transfers
type MemoryTransfer interface {
	// ToDevice transfers data from CPU to this device
	ToDevice(data []float64) []float64

	// ToCPU transfers data from this device to CPU
	ToCPU(data []float64) []float64
}

// ============================================================================
// Element-wise Operations Interface
// ============================================================================

type ElementWiseOps interface {
	// Basic arithmetic (in-place operations: a op b -> out)
	Add(a, b []float64, size int) []float64
	Sub(a, b []float64, size int) []float64
	Mul(a, b []float64, size int) []float64
	Div(a, b []float64, size int) []float64
	Neg(a []float64, size int) []float64

	// Scalar operations
	AddScalar(a []float64, scalar float64, size int) []float64
	SubScalar(a []float64, scalar float64, size int) []float64
	MulScalar(a []float64, scalar float64, size int) []float64
	DivScalar(a []float64, scalar float64, size int) []float64
}

// MathOps is an optional interface for advanced math functions
type MathOps interface {
	Exp(a []float64, size int) []float64
	Log(a []float64, size int) []float64
	Sqrt(a []float64, size int) []float64
	Pow(a []float64, power float64, size int) []float64
}

// ============================================================================
// Matrix Operations Interface
// ============================================================================

type MatrixOps interface {
	// MatMul performs matrix multiplication: C = A @ B
	// a: data buffer for matrix A with shape [m, k]
	// b: data buffer for matrix B with shape [k, n]
	// m, n, k: matrix dimensions
	MatMul(a, b, out []float64, m, n, k, strideA, strideB int) []float64

	// MatMulAdd performs matrix multiplication and addition: C = A @ B + C
	MatMulAdd(a, b, c []float64, m, n, k, strideA, strideB int) []float64

	// MatMulTransA performs matrix multiplication with A transposed: C = A^T @ B
	MatMulTransA(a, b, out []float64, m, n, k, strideA, strideB int) []float64

	// MatMulTransB performs matrix multiplication with B transposed: C = A @ B^T
	MatMulTransB(a, b, out []float64, m, n, k, strideA, strideB int) []float64
}

// BatchedMatrixOps is an optional interface for optimized batched operations
type BatchedMatrixOps interface {
	// BatchedMatMul performs batched matrix multiplication
	// a: [batchSize, m, k], b: [batchSize, k, n], out: [batchSize, m, n]
	BatchedMatMul(
		a, b []float64,
		batchSize, m, n, k int,
		transA, transB bool,
	) []float64
}

// ============================================================================
// Reduction Operations Interface
// ============================================================================

type ReductionOps interface {
	// Sum computes the sum of all elements
	Sum(a []float64, size int) float64

	// Mean computes the mean of all elements
	Mean(a []float64, size int) float64

	// Max finds the maximum element
	Max(a []float64, size int) float64

	// Min finds the minimum element
	Min(a []float64, size int) float64
}

// AxisReductionOps is an optional interface for axis-wise reductions
type AxisReductionOps interface {
	// SumAxis computes sum along specified axis
	SumAxis(a []float64, shape []int, axis int) []float64

	// MaxAxis computes max along specified axis
	MaxAxis(a []float64, shape []int, axis int) []float64

	// MeanAxis computes mean along specified axis
	MeanAxis(a []float64, shape []int, axis int) []float64
}

// ============================================================================
// Activation Functions Interface (Optional)
// ============================================================================

type ActivationOps interface {
	// ReLU applies ReLU activation: out = max(0, a)
	ReLU(a []float64, size int) []float64

	// ReLUBackward computes ReLU gradient
	ReLUBackward(grad, input []float64, size int) []float64

	// Sigmoid applies sigmoid activation
	Sigmoid(a []float64, size int) []float64

	// SigmoidBackward computes sigmoid gradient
	SigmoidBackward(grad, output []float64, size int) []float64

	// Tanh applies tanh activation
	Tanh(a []float64, size int) []float64

	// TanhBackward computes tanh gradient
	TanhBackward(grad, output []float64, size int) []float64

	// Exp computes the exponential
	Exp(a []float64, size int) []float64

	// Log computes the natural logarithm
	Log(a []float64, size int) []float64

	// Square computes the square
	Square(a []float64, size int) []float64
}

// SoftmaxOps is an optional interface for softmax operations
type SoftmaxOps interface {
	// Softmax applies softmax along the last dimension
	Softmax(data []float64, shape []int) []float64

	// SoftmaxBackward computes softmax gradient
	SoftmaxBackward(grad, output []float64, shape []int) []float64

	// LogSoftmax applies log(softmax(x)) - more numerically stable
	LogSoftmax(data []float64, shape []int) []float64
}

// ============================================================================
// Broadcasting Operations Interface (Optional)
// ============================================================================

type BroadcastOps interface {
	// BroadcastAdd performs broadcasted addition
	// a, b: input data buffers
	// aShape, bShape: shapes of a and b
	// outShape: resulting broadcasted shape
	BroadcastAdd(a, b []float64, aShape, bShape, outShape []int) []float64

	// BroadcastSub performs broadcasted subtraction
	BroadcastSub(a, b []float64, aShape, bShape, outShape []int) []float64

	// BroadcastMul performs broadcasted multiplication
	BroadcastMul(a, b []float64, aShape, bShape, outShape []int) []float64

	// BroadcastDiv performs broadcasted division
	BroadcastDiv(a, b []float64, aShape, bShape, outShape []int) []float64
}

// ============================================================================
// Utility Operations Interface (Optional)
// ============================================================================

type UtilityOps interface {
	// Fill sets all elements to a constant value
	Fill(data []float64, value float64, size int)

	// Clone creates a deep copy
	Clone(data []float64, size int) []float64

	// Transpose transposes a 2D matrix
	Transpose(a []float64, rows, cols int) []float64
}

// ============================================================================
// Helper Functions for Type Assertions
// ============================================================================

// SupportsMemoryTransfer checks if backend supports device transfers
func SupportsMemoryTransfer(b Backend) bool {
	_, ok := b.(MemoryTransfer)
	return ok
}

// SupportsBatchedMatMul checks if backend supports batched matrix multiplication
func SupportsBatchedMatMul(b Backend) bool {
	_, ok := b.(BatchedMatrixOps)
	return ok
}

// SupportsActivations checks if backend implements activation functions
func SupportsActivations(b Backend) bool {
	_, ok := b.(ActivationOps)
	return ok
}

// SupportsConvolutions checks if backend implements convolution operations
func SupportsConvolutions(b Backend) bool {
	_, ok := b.(ConvOps)
	return ok
}

// SupportsBroadcasting checks if backend implements broadcasting
func SupportsBroadcasting(b Backend) bool {
	_, ok := b.(BroadcastOps)
	return ok
}

// SupportsSoftmax checks if backend implements softmax operations
func SupportsSoftmax(b Backend) bool {
	_, ok := b.(SoftmaxOps)
	return ok
}

// SupportsAxisReductions checks if backend implements axis-wise reductions
func SupportsAxisReductions(b Backend) bool {
	_, ok := b.(AxisReductionOps)
	return ok
}

// ============================================================================
// Backend Registry
// ============================================================================

var (
	defaultBackend Backend
	backends       = make(map[string]Backend)
)

// RegisterBackend registers a backend with a name
func RegisterBackend(name string, backend Backend) {
	backends[name] = backend
}

// GetBackend retrieves a backend by name
func GetBackend(name string) (Backend, bool) {
	b, ok := backends[name]
	return b, ok
}

// SetDefaultBackend sets the default backend
func SetDefaultBackend(backend Backend) {
	defaultBackend = backend
}

// GetDefaultBackend returns the default backend
func GetDefaultBackend() Backend {
	return defaultBackend
}

// ListBackends returns all registered backend names
func ListBackends() []string {
	names := make([]string, 0, len(backends))
	for name := range backends {
		names = append(names, name)
	}
	return names
}

// ============================================================================
// Automatic Backend Selection
// ============================================================================

// AutoSelectBackend automatically selects the best available backend
func AutoSelectBackend() Backend {
	// Try CUDA first
	if b, ok := backends["cuda"]; ok {
		return b
	}

	// Try ROCm
	if b, ok := backends["rocm"]; ok {
		return b
	}

	// Fall back to CPU
	if b, ok := backends["cpu"]; ok {
		return b
	}

	panic("no backend available")
}

// ============================================================================
// Base Implementation
// ============================================================================

// Base provides common functionality for all backends
type Base struct {
	name  string
	isGPU bool
}

func (b *Base) Name() string {
	return b.name
}

func (b *Base) IsGPU() bool {
	return b.isGPU
}

// NewBase is a small helper for creating a Base
func NewBase(name string, isGPU bool) *Base {
	return &Base{name: name, isGPU: isGPU}
}
