package tensor

import (
	"math"

	"github.com/kabironline/nanograd/backend"
)

type Tensor struct {
	data    *Storage
	grad    *Storage
	Shape   []int
	Strides []int
	Parents []*Tensor // Tensors that were used to compute this tensor

	Backward func() // Function to compute gradients during backpropagation
	Offset   int    // Starting point in the Data slice

	Device backend.Backend

	RequiresGrad bool

	contiguous bool
}

func NewTensor(data []float32, shape []int, parents ...*Tensor) *Tensor {
	var dev backend.Backend
	requiresGrad := false
	if len(parents) > 0 {
		dev = parents[0].Device
		for _, p := range parents {
			if p.RequiresGrad {
				requiresGrad = true
				break
			}
		}
	}
	if dev == nil {
		dev = backend.AutoSelectBackend()
	}

	// If backend is GPU, move data to device
	if dev.IsGPU() {
		if memTransfer, ok := dev.(backend.MemoryTransfer); ok {
			data = memTransfer.ToDevice(data)
		}
	}

	dataStorage := StorageFrom(data)

	return &Tensor{
		data:    dataStorage,
		Shape:   shape,
		Strides: computeStrides(shape),
		// grad is allocated lazily to avoid unnecessary allocations for tensors
		// that never participate in backpropagation.
		grad:         nil,
		Parents:      parents,
		Device:       dev,
		RequiresGrad: requiresGrad,
		contiguous:   true,
	}
}

// NewEmptyTensor creates a new tensor with the given shape and data pre-allocated by the backend
func NewEmptyTensor(shape []int, dev backend.Backend) *Tensor {
	if dev == nil {
		dev = backend.AutoSelectBackend()
	}
	size := 1
	for _, s := range shape {
		size *= s
	}

	// Bridge: keep using the []float32 Allocate so the F32 compute path (and the
	// CUDA fake-device-slice) keep working. Switch to dev.AllocStorage once the
	// compute ops take *Storage (P5+) and no longer call F32() on device memory.
	dataStorage := StorageFrom(dev.Allocate(size))

	return &Tensor{
		data:       dataStorage,
		Shape:      shape,
		Strides:    computeStrides(shape),
		Device:     dev,
		contiguous: true,
	}
}

// FromData wraps an existing buffer into a contiguous Tensor WITHOUT a host->device
// copy. It is the exported way for packages outside `tensor` (e.g. nn layers) to
// build a Tensor from a result buffer returned by a backend op and attach autograd
// metadata; set Backward on the result afterward. Strides are row-major for shape.
func FromData(data []float32, shape []int, dev backend.Backend, requiresGrad bool, parents ...*Tensor) *Tensor {
	return &Tensor{
		data:         StorageFrom(data),
		Shape:        shape,
		Strides:      ComputeStrides(shape),
		Device:       dev,
		RequiresGrad: requiresGrad,
		Parents:      parents,
		contiguous:   true,
	}
}

// initializes an identity matrix tensor of given size (size x size)
func NewIdentityTensor(size int) *Tensor {
	dev := backend.AutoSelectBackend()

	// For GPU backends, create on CPU then copy to device
	// TODO: IMPLMENET LATER
	// if dev.IsGPU() {
	// 	if transfer, ok := dev.(backend.MemoryTransfer); ok {
	// 		h_data := make([]float32, size*size)
	// 		for i := 0; i < size; i++ {
	// 			h_data[i*size+i] = 1.0
	// 		}
	// 		data := transfer.ToDevice(h_data)
	// 		return &Tensor{
	// 			data:    data,
	// 			Shape:   []int{size, size},
	// 			Strides: []int{size, 1},
	// 			grad:         nil,
	// 			Device:       dev,
	// 			contiguous:   true,
	// 		}
	// 	}
	// }

	// For CPU backend (or GPU without MemoryTransfer), allocate and initialize directly
	data := dev.Allocate(size * size)
	for i := 0; i < size; i++ {
		data[i*size+i] = 1.0
	}

	dataStorage := StorageFrom(data)

	return &Tensor{
		data:       dataStorage,
		Shape:      []int{size, size},
		Strides:    []int{size, 1},
		grad:       nil,
		Device:     dev,
		contiguous: true,
	}
}

// Data and Grad are transitional shims that expose the underlying storage as a
// []float32 so existing F32-only call sites keep working. DELETE in Phase 6.
// Grad returns nil (rather than panicking) when no gradient is allocated yet,
// so `t.Grad() == nil` remains a valid "is grad allocated?" check.
func (t *Tensor) Data() []float32 {
	if t.data == nil {
		return nil
	}
	return t.data.F32()
}
func (t *Tensor) Grad() []float32 {
	if t.grad == nil {
		return nil
	}
	return t.grad.F32()
}

// Helper to calculate strides for row-major order
func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return strides
}

// RandomInit fills the tensor with Xavier/Glorot initialization
func (t *Tensor) RandomInit() {
	// Glorot / Xavier normal initialization:
	// draw from N(0, stdDev^2) where stdDev = sqrt(2 / (fanIn + fanOut))
	var fanIn, fanOut int

	if len(t.Shape) == 0 {
		// fallback: use total elements for both
		fanIn = t.data.Length()
		fanOut = t.data.Length()
	} else if len(t.Shape) == 1 {
		// vector: treat as both in and out
		fanIn = t.Shape[0]
		fanOut = t.Shape[0]
	} else {
		// For typical weight tensors:
		// shape[0] = out, shape[1] = in, remaining dims are kernel dimensions (for conv)
		kernel := 1
		if len(t.Shape) > 2 {
			for i := 2; i < len(t.Shape); i++ {
				kernel *= t.Shape[i]
			}
		}
		fanIn = t.Shape[1] * kernel
		fanOut = t.Shape[0] * kernel
	}

	if fanIn <= 0 {
		fanIn = 1
	}
	if fanOut <= 0 {
		fanOut = 1
	}

	stdDev := float32(math.Sqrt(2.0 / float64(fanIn+fanOut)))
	t.Device.Normal(t.Data(), 0.0, stdDev, t.data.Length())
}

// ZeroInit fills the tensor with zeros
func (t *Tensor) ZeroInit() {
	t.Device.Fill(t.Data(), 0.0, t.data.Length())
}

// AccumulateGrad adds the given gradient data to the tensor's gradient.
// It handles non-contiguous tensors (views) correctly by mapping logical indices to physical indices.
// grad must be a contiguous slice of data matching the logical shape of the tensor.
func (t *Tensor) AccumulateGrad(grad []float32) {
	if !t.RequiresGrad {
		return
	}
	if len(grad) != TotalSize(t.Shape) {
		panic("AccumulateGrad: gradient size does not match tensor shape")
	}

	// Ensure a grad buffer exists for accumulation.
	t.ensureGrad()

	// Fast path: contiguous tensors use backend Add operation
	if t.Contiguous() && t.Offset == 0 {
		// Use backend's Add operation: t.Grad = t.Grad + grad
		t.Device.Add(t.Grad(), grad, t.Grad(), t.grad.Length())
		return
	}

	// Medium-fast path: aligned gradients
	if len(grad) == t.grad.Length() && t.Offset == 0 {
		t.Device.Add(t.Grad(), grad, t.Grad(), t.grad.Length())
		return
	}

	// Slow path for views - need to copy to CPU for indexed access
	// This is inefficient for GPU but views should be rare during backprop
	if t.Device.IsGPU() {
		if memTransfer, ok := t.Device.(backend.MemoryTransfer); ok {
			// Copy gradients to CPU
			cpuGrad := memTransfer.ToCPU(t.Grad())
			cpuIncomingGrad := memTransfer.ToCPU(grad)

			// Accumulate on CPU
			for i, g := range cpuIncomingGrad {
				physicalIdx := t.PhysicalIndexFromLinearIndex(i)
				cpuGrad[physicalIdx] += g
			}

			// Copy back to GPU
			t.grad = StorageFrom(memTransfer.ToDevice(cpuGrad))
			return
		}
	}

	// CPU fallback for views
	for i, g := range grad {
		physicalIdx := t.PhysicalIndexFromLinearIndex(i)
		t.Grad()[physicalIdx] += g
	}
}

// TotalSize computes the total number of elements of a tensor from its shape.
func (t *Tensor) TotalSize() int {
	total := 1
	for _, dim := range t.Shape {
		total *= dim
	}
	return total
}

// Contiguous returns whether the tensor's storage is contiguous in row-major order.
func (t *Tensor) Contiguous() bool {
	return t.contiguous
}

// Free releases GPU memory for this tensor's data and gradients.
// Only call this when the tensor is no longer needed.
// WARNING: Do not use the tensor after calling Free()!
func (t *Tensor) Free() {
	if t.data != nil && t.Device.IsGPU() {
		t.Device.Free(t.Data())
		t.data = nil
	}
	if t.grad != nil && t.Device.IsGPU() {
		t.Device.Free(t.Grad())
		t.grad = nil
	}
}

// ClearGraph breaks references to parent tensors and backward functions,
// allowing Go's GC to collect intermediate tensors.
// Call this after backward pass completes to prevent memory leaks.
func (t *Tensor) ClearGraph() {
	t.Parents = nil
	t.Backward = nil
	// Free GPU memory for Data and Grad
	if t.data != nil && t.Device.IsGPU() {
		t.Device.Free(t.Data())
		t.data = nil
	}
	if t.grad != nil && t.Device.IsGPU() {
		t.Device.Free(t.Grad())
		t.grad = nil
	}
}

// ClearComputationGraph clears the entire computation graph starting from this tensor.
// It walks the graph and clears all non-parameter tensors (tensors that had Parents).
// Parameters (leaf tensors with RequiresGrad but no original Parents) are preserved.
// This should be called on the loss tensor after backward completes.
func (t *Tensor) ClearComputationGraph() {
	// Use a set to track visited tensors and avoid cycles
	visited := make(map[*Tensor]bool)
	t.clearGraphHelper(visited)
}

func (t *Tensor) clearGraphHelper(visited map[*Tensor]bool) {
	if t == nil || visited[t] {
		return
	}
	visited[t] = true

	// Save the parents list before clearing
	parents := t.Parents

	// Don't clear leaf parameters (RequiresGrad but originally had no parents)
	// We identify these as tensors with RequiresGrad=true
	// If a tensor has RequiresGrad but is also in this graph, it means it was created
	// during forward pass (like intermediate activations that require grad).
	// Parameters are created separately and should have been leaf nodes.
	//
	// A better heuristic: if this tensor has no Parents NOW, it's either:
	// 1. A parameter (leaf node) - DON'T clear its data
	// 2. Already been cleared - skip
	//
	// If it DOES have Parents, it's an intermediate tensor - clear everything

	if len(parents) > 0 {
		// Intermediate tensor - recursively visit parents first
		for _, parent := range parents {
			parent.clearGraphHelper(visited)
		}

		// Then clear this intermediate tensor completely
		t.ClearGraph()
	} else {
		// Leaf tensor (parameter or already cleared) - only clear backward/parents refs
		// DON'T free Data or Grad - parameters need both for the next iteration
		t.Parents = nil
		t.Backward = nil
	}
}

// ensureGrad makes sure t.Grad is allocated. It's safe to call multiple times.
func (t *Tensor) ensureGrad() {
	if t.grad == nil {
		// Use underlying data length to ensure we can index into physical positions
		t.grad = StorageFrom(t.Device.Allocate(t.data.Length()))
		// Initialize gradient to zero
		t.Device.Fill(t.Grad(), 0, t.grad.Length())
	}
}

// ZeroGrad zeroes this tensor's gradient buffer, allocating it first if it does
// not exist yet. It is a no-op for tensors that do not require gradients.
//
// Optimizers zero only the parameters registered with them; use this when you
// need to reset a specific tensor (e.g. between two independent backward passes
// over the same leaf, as gradient checking does).
func (t *Tensor) ZeroGrad() {
	if !t.RequiresGrad {
		return
	}
	t.ensureGrad()
	t.Device.Fill(t.Grad(), 0, t.grad.Length())
}

// AllocGrad ensures the tensor has a gradient buffer allocated. Public helper
// for packages that create parameters which will always participate in
// backpropagation (e.g., model weights and biases).
func (t *Tensor) AllocGrad() {
	if t.grad == nil {
		t.grad = StorageFrom(t.Device.Allocate(TotalSize(t.Shape)))
		t.Device.Fill(t.Grad(), 0, t.grad.Length())
	}
}

// ToGradTensor returns a new Tensor sharing the Grad data of this tensor.
// Useful for autograd where we need to perform tensor operations on gradients.
func (t *Tensor) ToGradTensor() *Tensor {
	if t.grad == nil {
		t.AllocGrad()
	}
	return &Tensor{
		data:         t.grad,
		Shape:        append([]int(nil), t.Shape...),
		Strides:      append([]int(nil), t.Strides...),
		Offset:       t.Offset,
		Device:       t.Device,
		RequiresGrad: false, // Gradients do not require gradients themselves by default
	}
}

// To moves the tensor to the specified device.
func (t *Tensor) To(dev backend.Backend) *Tensor {
	if t.Device == dev {
		return t
	}

	// Transfer data
	if transfer, ok := dev.(backend.MemoryTransfer); ok {
		t.data = StorageFrom(transfer.ToDevice(t.data.F32()))
		if t.grad != nil {
			t.grad = StorageFrom(transfer.ToDevice(t.grad.F32()))
		}
	} else {
		// Fallback: copy via CPU if possible, or just copy data
		newData := StorageFrom(dev.Allocate(t.data.Length()))
		dev.Copy(newData.F32(), t.data.F32())
		t.data = newData
		if t.grad != nil {
			newGrad := StorageFrom(dev.Allocate(t.grad.Length()))
			dev.Copy(newGrad.F32(), t.grad.F32())
			t.grad = newGrad
		}
	}

	t.Device = dev
	return t
}
