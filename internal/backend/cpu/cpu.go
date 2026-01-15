package cpu

import (
	"runtime"

	"github.com/kabironline/nanograd/backend"
)

// CPUBackend implements the Backend interface for CPU computations.
// It also implements optional interfaces: ActivationOps, BroadcastOps, SoftmaxOps, MemoryTransfer
type CPUBackend struct {
	backend.Base
	numWorkers        int
	parallelThreshold int
	pool              *WorkerPool
}

// NewCPUBackend creates and initializes a new CPU backend
func NewCPUBackend() *CPUBackend {
	numWorkers := runtime.NumCPU()
	return &CPUBackend{
		Base:              *backend.NewBase("cpu", false),
		numWorkers:        numWorkers,
		parallelThreshold: 4096,
		pool:              NewWorkerPool(numWorkers),
	}
}

func init() {
	backend.RegisterBackend("cpu", NewCPUBackend())
}

func (b *CPUBackend) Name() string {
	return "cpu"
}

func (b *CPUBackend) IsGPU() bool {
	return false
}

// ============================================================================
// Memory Management (required)
// ============================================================================

// Allocate creates a new buffer of the given size on the CPU.
func (b *CPUBackend) Allocate(size int) []float32 {
	return make([]float32, size)
}

// Free releases device memory (no-op for CPU).
func (b *CPUBackend) Free(data []float32) {
	// No action needed for CPU memory - handled by Go GC
}

// Copy performs a device-to-device copy on CPU.
func (b *CPUBackend) Copy(dst, src []float32) {
	copy(dst, src)
}

// ============================================================================
// Memory Transfer (optional interface)
// ============================================================================

// ToDevice transfers data from CPU to this device (no-op for CPU).
func (b *CPUBackend) ToDevice(data []float32) []float32 {
	return data
}

// ToCPU transfers data from this device to CPU (no-op for CPU).
func (b *CPUBackend) ToCPU(data []float32) []float32 {
	return data
}
