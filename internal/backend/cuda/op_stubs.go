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

func (b *CUDABackend) Neg(a, out []float32, size int) { cpu.NewCPUBackend().Neg(a, out, size) }
func (b *CUDABackend) AddScalar(a []float32, scalar float32, size int) []float32 {
	hA := b.ToCPU(a)
	b.Sync()
	hOut := cpuBackend().AddScalar(hA, scalar, size)
	dOut := b.ToDevice(hOut)
	b.Sync()
	return dOut
}
func (b *CUDABackend) SubScalar(a []float32, scalar float32, size int) []float32 {
	hA := b.ToCPU(a)
	b.Sync()
	hOut := cpuBackend().SubScalar(hA, scalar, size)
	dOut := b.ToDevice(hOut)
	b.Sync()
	return dOut
}
func (b *CUDABackend) MulScalar(a []float32, scalar float32, size int) []float32 {
	hA := b.ToCPU(a)
	b.Sync()
	hOut := cpuBackend().MulScalar(hA, scalar, size)
	dOut := b.ToDevice(hOut)
	b.Sync()
	return dOut
}
func (b *CUDABackend) DivScalar(a []float32, scalar float32, size int) []float32 {
	hA := b.ToCPU(a)
	b.Sync()
	hOut := cpuBackend().DivScalar(hA, scalar, size)
	dOut := b.ToDevice(hOut)
	b.Sync()
	return dOut
}

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
func (b *CUDABackend) Exp(a []float32, size int) []float32 { return cpuBackend().Exp(a, size) }
func (b *CUDABackend) Log(a []float32, size int) []float32 { return cpuBackend().Log(a, size) }
func (b *CUDABackend) Sqrt(a []float32, size int) []float32 {
	hA := b.ToCPU(a)
	b.Sync()
	out := cpuBackend().Allocate(size)
	for i := 0; i < size; i++ {
		out[i] = float32(math.Sqrt(float64(hA[i])))
	}
	dOut := b.ToDevice(out)
	b.Sync()
	return dOut
}
func (b *CUDABackend) Square(a []float32, size int) []float32 { return cpuBackend().Square(a, size) }

// Matrix Operations
// func (b *CUDABackend) MatMul(a, b_val []float32, m, n, k, strideA, strideB int) []float32 { return nil }
func (b *CUDABackend) MatMulAdd(a, b_val, c, out []float32, m, n, k, strideA, strideB int) {
	cpuBackend().MatMulAdd(a, b_val, c, out, m, n, k, strideA, strideB)
}
func (b *CUDABackend) MatMulTransA(a, b_val, out []float32, m, n, k, strideA, strideB int) []float32 {
	return cpu.NewCPUBackend().MatMulTransA(a, b_val, out, m, n, k, strideA, strideB)
}
func (b *CUDABackend) MatMulTransB(a, b_val, out []float32, m, n, k, strideA, strideB int) []float32 {
	return cpu.NewCPUBackend().MatMulTransB(a, b_val, out, m, n, k, strideA, strideB)
}

// Reduction Operations
func (b *CUDABackend) Sum(data []float32, size int) float32 {
	h := b.ToCPU(data)
	b.Sync()
	return cpuBackend().Sum(h, size)
}
func (b *CUDABackend) Mean(data []float32, size int) float32 {
	h := b.ToCPU(data)
	b.Sync()
	return cpuBackend().Mean(h, size)
}
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
func (b *CUDABackend) SumAxis(data []float32, shape []int, axis int) []float32 {
	h := b.ToCPU(data)
	b.Sync()
	hOut := cpuBackend().SumAxis(h, shape, axis)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
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

// Conv
func (b *CUDABackend) Im2Col(data []float32, shape, strides []int, kH, kW, stride, padding int) []float32 {
	h := b.ToCPU(data)
	b.Sync()
	hOut := cpuBackend().Im2Col(h, shape, strides, kH, kW, stride, padding)
	d := b.ToDevice(hOut)
	b.Sync()
	return d
}
func (b *CUDABackend) Col2Im(colGrad []float32, xShape, xStrides []int, kH, kW, stride, padding int) []float32 {
	h := b.ToCPU(colGrad)
	b.Sync()
	hOut := cpuBackend().Col2Im(h, xShape, xStrides, kH, kW, stride, padding)
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
	h := b.ToCPU(data)
	b.Sync()
	cpuBackend().Normal(h, mean, stdDev, size)
	tmp := b.ToDevice(h)
	b.Copy(data, tmp)
	b.Free(tmp)
	b.Sync()
}

// Optimizer
func (b *CUDABackend) StepSGD(data, grad []float32, lr float32) {
	hData := b.ToCPU(data)
	hGrad := b.ToCPU(grad)
	b.Sync()
	cpuBackend().StepSGD(hData, hGrad, lr)
	tmp := b.ToDevice(hData)
	b.Copy(data, tmp)
	b.Free(tmp)
	b.Sync()
}
func (b *CUDABackend) StepAdam(data, grad, m, v []float32, lr, beta1, beta2, eps float32, t int) {
	hData := b.ToCPU(data)
	hGrad := b.ToCPU(grad)
	hm := b.ToCPU(m)
	hv := b.ToCPU(v)
	b.Sync()
	cpuBackend().StepAdam(hData, hGrad, hm, hv, lr, beta1, beta2, eps, t)
	tmp := b.ToDevice(hData)
	b.Copy(data, tmp)
	b.Free(tmp)
	tmpm := b.ToDevice(hm)
	b.Copy(m, tmpm)
	b.Free(tmpm)
	tmpv := b.ToDevice(hv)
	b.Copy(v, tmpv)
	b.Free(tmpv)
	b.Sync()
}
