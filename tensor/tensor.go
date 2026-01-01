package tensor

import (
	"math"
	"math/rand"
)

type Tensor struct {
	Data    []float64
	Shape   []int
	Strides []int
	Offset  int // Starting point in the Data slice
	Grad    []float64
	Parents []*Tensor // Tensors that were used to compute this tensor

	Backward func() // Function to compute gradients during backpropagation
}

func NewTensor(data []float64, shape []int, parents ...*Tensor) *Tensor {
	return &Tensor{
		Data:    data,
		Shape:   shape,
		Strides: defaultStrides(shape),
		Grad:    make([]float64, len(data)),
		Parents: parents,
	}
}

// initializes an identity matrix tensor of given size (size x size)
func NewIdentityTensor(size int) *Tensor {
	data := make([]float64, size*size)
	for i := range size {
		data[i*size+i] = 1.0
	}
	return &Tensor{
		Data:    data,
		Shape:   []int{size, size},
		Strides: []int{size, 1},
		Grad:    make([]float64, size*size),
	}
}

// Helper to calculate strides for row-major order
func defaultStrides(shape []int) []int {
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
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64() * stdDev
	}
}

// ZeroInit fills the tensor with zeros
func (t *Tensor) ZeroInit() {
	for i := range t.Data {
		t.Data[i] = 0.0
	}
}

// AccumulateGrad adds the given gradient data to the tensor's gradient.
// It handles non-contiguous tensors (views) correctly by mapping logical indices to physical indices.
// grad must be a contiguous slice of data matching the logical shape of the tensor.
func (t *Tensor) AccumulateGrad(grad []float64) {
	if len(grad) != TotalSize(t.Shape) {
		panic("AccumulateGrad: gradient size does not match tensor shape")
	}

	// Optimization for contiguous tensors
	if IsContiguous(t.Shape, t.Strides) && t.Offset == 0 {
		for i, g := range grad {
			t.Grad[i] += g
		}
		return
	}

	// Slow path for views
	// The input 'grad' is assumed to be contiguous and row-major, so we use default strides for it.
	gradStrides := defaultStrides(t.Shape)
	for i, g := range grad {
		coords := CoordsFromLinearIndex(i, t.Shape, gradStrides)
		physicalIdx := LinearIndexFromCoords(coords, t.Shape, t.Strides) + t.Offset
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
