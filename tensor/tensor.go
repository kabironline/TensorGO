package tensor

import (
	"math"

	"github.com/kabironline/nanograd/backend"
	"gonum.org/v1/gonum/floats"
)

type Tensor struct {
	Data    []float64
	Grad    []float64
	Shape   []int
	Strides []int
	Parents []*Tensor // Tensors that were used to compute this tensor

	Backward func() // Function to compute gradients during backpropagation
	Offset   int    // Starting point in the Data slice

	Device backend.Backend

	RequiresGrad bool

	contiguous bool
}

func NewTensor(data []float64, shape []int, parents ...*Tensor) *Tensor {
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

	return &Tensor{
		Data:    data,
		Shape:   shape,
		Strides: computeStrides(shape),
		// Grad is allocated lazily to avoid unnecessary allocations for tensors
		// that never participate in backpropagation.
		Grad:         nil,
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
	return &Tensor{
		Data:       dev.Allocate(size),
		Shape:      shape,
		Strides:    computeStrides(shape),
		Device:     dev,
		contiguous: true,
	}
}

// initializes an identity matrix tensor of given size (size x size)
func NewIdentityTensor(size int) *Tensor {
	dev := backend.AutoSelectBackend()

	// For GPU backends, create on CPU then copy to device
	if dev.IsGPU() {
		if transfer, ok := dev.(backend.MemoryTransfer); ok {
			h_data := make([]float64, size*size)
			for i := 0; i < size; i++ {
				h_data[i*size+i] = 1.0
			}
			data := transfer.ToDevice(h_data)
			return &Tensor{
				Data:       data,
				Shape:      []int{size, size},
				Strides:    []int{size, 1},
				Grad:       nil,
				Device:     dev,
				contiguous: true,
			}
		}
	}

	// For CPU backend (or GPU without MemoryTransfer), allocate and initialize directly
	data := dev.Allocate(size * size)
	for i := 0; i < size; i++ {
		data[i*size+i] = 1.0
	}
	return &Tensor{
		Data:       data,
		Shape:      []int{size, size},
		Strides:    []int{size, 1},
		Grad:       nil,
		Device:     dev,
		contiguous: true,
	}
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
		fanIn = len(t.Data)
		fanOut = len(t.Data)
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

	stdDev := math.Sqrt(2.0 / float64(fanIn+fanOut))
	t.Device.Normal(t.Data, 0.0, stdDev, len(t.Data))
}

// ZeroInit fills the tensor with zeros
func (t *Tensor) ZeroInit() {
	t.Device.Fill(t.Data, 0.0, len(t.Data))
}

// AccumulateGrad adds the given gradient data to the tensor's gradient.
// It handles non-contiguous tensors (views) correctly by mapping logical indices to physical indices.
// grad must be a contiguous slice of data matching the logical shape of the tensor.
func (t *Tensor) AccumulateGrad(grad []float64) {
	if !t.RequiresGrad {
		return
	}
	if len(grad) != TotalSize(t.Shape) {
		panic("AccumulateGrad: gradient size does not match tensor shape")
	}

	// Ensure a grad buffer exists for accumulation.
	t.ensureGrad()

	// Fast path: contiguous tensors use SIMD
	if t.Contiguous() && t.Offset == 0 {
		floats.Add(t.Grad, grad)
		return
	}

	// Medium-fast path: aligned gradients
	if len(grad) == len(t.Grad) && t.Offset == 0 {
		for i := range grad {
			t.Grad[i] += grad[i]
		}
		return
	}

	// Slow path for views
	for i, g := range grad {
		physicalIdx := t.PhysicalIndexFromLinearIndex(i)
		t.Grad[physicalIdx] += g
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

// ensureGrad makes sure t.Grad is allocated. It's safe to call multiple times.
func (t *Tensor) ensureGrad() {
	if t.Grad == nil {
		// Use underlying data length to ensure we can index into physical positions
		t.Grad = t.Device.Allocate(len(t.Data))
	}
}

// AllocGrad ensures the tensor has a gradient buffer allocated. Public helper
// for packages that create parameters which will always participate in
// backpropagation (e.g., model weights and biases).
func (t *Tensor) AllocGrad() {
	if t.Grad == nil {
		t.Grad = t.Device.Allocate(TotalSize(t.Shape))
		t.Device.Fill(t.Grad, 0, len(t.Grad))
	}
}

// ToGradTensor returns a new Tensor sharing the Grad data of this tensor.
// Useful for autograd where we need to perform tensor operations on gradients.
func (t *Tensor) ToGradTensor() *Tensor {
	if t.Grad == nil {
		t.AllocGrad()
	}
	return &Tensor{
		Data:         t.Grad,
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
		t.Data = transfer.ToDevice(t.Data)
		if t.Grad != nil {
			t.Grad = transfer.ToDevice(t.Grad)
		}
	} else {
		// Fallback: copy via CPU if possible, or just copy data
		newData := dev.Allocate(len(t.Data))
		dev.Copy(newData, t.Data)
		t.Data = newData
		if t.Grad != nil {
			newGrad := dev.Allocate(len(t.Grad))
			dev.Copy(newGrad, t.Grad)
			t.Grad = newGrad
		}
	}

	t.Device = dev
	return t
}
