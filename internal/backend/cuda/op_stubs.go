package cuda

import (
	"math"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cpu"
)

func cpuBackend() *cpu.CPUBackend {
	if b, ok := backend.GetBackend("cpu"); ok {
		if cb, ok := b.(*cpu.CPUBackend); ok {
			return cb
		}
	}
	return cpu.NewCPUBackend()
}

// ============================================================================
// Element-wise Operations (Minimal set to satisfy interface)
// ============================================================================

// Scalar operations now have CUDA implementations in ops_scalar.go
// (AddScalar, SubScalar, MulScalar, DivScalar)

func (b *CUDABackend) Pow(a []float32, power float32, size int) []float32 {
	hA := b.ToCPU(a)
	b.Sync()
	out := cpuBackend().Allocate(size)
	for i := 0; i < size; i++ {
		out[i] = float32(math.Pow(float64(hA[i]), float64(power)))
	}
	dOut := b.ToDevice(out)
	b.Sync()
	return dOut
}

// Matrix Operations
//
// MatMulAdd was removed with the MatOperand migration: accumulation is now
// expressed by MatMul's beta parameter. The old stub was broken regardless — it
// forwarded a length-n bias to Add, which requires operands of equal length.

// Reduction Operations
// Sum, Mean, and SumAxis now have CUDA implementations in ops_reduction.go

func (b *CUDABackend) Max(data []float32, size int) float32 {
	h := b.ToCPU(data)
	b.Sync()
	return cpuBackend().Max(h, size)
}
func (b *CUDABackend) Min(data []float32, size int) float32 {
	h := b.ToCPU(data)
	b.Sync()
	return cpuBackend().Min(h, size)
}

// Axis Reductions
// SumAxis now has CUDA implementation in ops_reduction.go

func (b *CUDABackend) MeanAxis(data []float32, shape []int, axis int) []float32 {
	h := b.ToCPU(data)
	b.Sync()
	hOut := cpuBackend().MeanAxis(h, shape, axis)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) MaxAxis(data []float32, shape []int, axis int) []float32 {
	h := b.ToCPU(data)
	b.Sync()
	hOut := cpuBackend().MaxAxis(h, shape, axis)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}

// NN Ops
//
//	func (b *CUDABackend) ReLU(a []float32, size int) []float32 {
//		h := b.ToCPU(a)
//		b.Sync()
//		hOut := cpuBackend().ReLU(h, size)
//		d := b.ToDevice(hOut)
//		b.Sync()
//		return d
//	}
//
//	func (b *CUDABackend) ReLUBackward(grad, input []float32, size int) []float32 {
//		hG := b.ToCPU(grad)
//		hI := b.ToCPU(input)
//		b.Sync()
//		hOut := cpuBackend().ReLUBackward(hG, hI, size)
//		d := b.ToDevice(hOut)
//		b.Sync()
//		return d
//	}
func (b *CUDABackend) Sigmoid(a []float32, size int) []float32 {
	h := b.ToCPU(a)
	b.Sync()
	hOut := cpuBackend().Sigmoid(h, size)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) SigmoidBackward(grad, output []float32, size int) []float32 {
	hG := b.ToCPU(grad)
	hO := b.ToCPU(output)
	b.Sync()
	hOut := cpuBackend().SigmoidBackward(hG, hO, size)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) Tanh(a []float32, size int) []float32 {
	h := b.ToCPU(a)
	b.Sync()
	hOut := cpuBackend().Tanh(h, size)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) TanhBackward(grad, output []float32, size int) []float32 {
	hG := b.ToCPU(grad)
	hO := b.ToCPU(output)
	b.Sync()
	hOut := cpuBackend().TanhBackward(hG, hO, size)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) Softmax(data []float32, shape []int) []float32 {
	h := b.ToCPU(data)
	b.Sync()
	hOut := cpuBackend().Softmax(h, shape)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) SoftmaxBackward(grad, output []float32, shape []int) []float32 {
	hG := b.ToCPU(grad)
	hO := b.ToCPU(output)
	b.Sync()
	hOut := cpuBackend().SoftmaxBackward(hG, hO, shape)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) LogSoftmax(data []float32, shape []int) []float32 {
	h := b.ToCPU(data)
	b.Sync()
	hOut := cpuBackend().LogSoftmax(h, shape)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}

// Broadcast
func (b *CUDABackend) BroadcastAdd(a, b_val []float32, aShape, bShape, outShape []int) []float32 {
	hA := b.ToCPU(a)
	hB := b.ToCPU(b_val)
	b.Sync()
	hOut := cpuBackend().BroadcastAdd(hA, hB, aShape, bShape, outShape)
	dOut := b.ToDevice(hOut)
	b.Sync()
	return dOut
}
func (b *CUDABackend) BroadcastSub(a, b_val []float32, aShape, bShape, outShape []int) []float32 {
	hA := b.ToCPU(a)
	hB := b.ToCPU(b_val)
	b.Sync()
	hOut := cpuBackend().BroadcastSub(hA, hB, aShape, bShape, outShape)
	dOut := b.ToDevice(hOut)
	b.Sync()
	return dOut
}
func (b *CUDABackend) BroadcastMul(a, b_val []float32, aShape, bShape, outShape []int) []float32 {
	hA := b.ToCPU(a)
	hB := b.ToCPU(b_val)
	b.Sync()
	hOut := cpuBackend().BroadcastMul(hA, hB, aShape, bShape, outShape)
	dOut := b.ToDevice(hOut)
	b.Sync()
	return dOut
}
func (b *CUDABackend) BroadcastDiv(a, b_val []float32, aShape, bShape, outShape []int) []float32 {
	hA := b.ToCPU(a)
	hB := b.ToCPU(b_val)
	b.Sync()
	hOut := cpuBackend().BroadcastDiv(hA, hB, aShape, bShape, outShape)
	dOut := b.ToDevice(hOut)
	b.Sync()
	return dOut
}

// Utility
func (b *CUDABackend) Fill(data []float32, value float32, size int) {
	if size == 0 || len(data) == 0 {
		return
	}
	// Fill on the host and write back to device
	h := make([]float32, size)
	cpuBackend().Fill(h, value, size)
	b.WriteToDevice(data, h)
	b.Sync()
}
func (b *CUDABackend) Clone(data []float32, size int) []float32 {
	h := b.ToCPU(data)
	b.Sync()
	hOut := cpuBackend().Clone(h, size)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) Transpose(a []float32, rows, cols int) []float32 {
	h := b.ToCPU(a)
	b.Sync()
	hOut := cpuBackend().Transpose(h, rows, cols)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}

// Pool
func (b *CUDABackend) MaxPool2d(data []float32, shape, strides []int, kH, kW, stride, padding int) ([]float32, []int) {
	h := b.ToCPU(data)
	b.Sync()
	hOut, idx := cpuBackend().MaxPool2d(h, shape, strides, kH, kW, stride, padding)
	d := b.ToDevice(hOut)
	b.Sync()
	return d, idx
}
func (b *CUDABackend) MaxPool2dBackward(grad []float32, indices []int, xShape []int) []float32 {
	h := b.ToCPU(grad)
	b.Sync()
	hOut := cpuBackend().MaxPool2dBackward(h, indices, xShape)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}

// Random
func (b *CUDABackend) Normal(data []float32, mean, stdDev float32, size int) {
	// Generate random numbers on CPU
	h := make([]float32, size)
	cpuBackend().Normal(h, mean, stdDev, size)

	// Copy to device memory (data is expected to be device memory)
	b.WriteToDevice(data, h)
	b.Sync()
}
