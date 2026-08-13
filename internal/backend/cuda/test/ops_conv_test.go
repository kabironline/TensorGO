//go:build cuda

package cuda_test

import (
	"math/rand/v2"
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCudaConv2DForwardMatchesCPU(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	require.NoError(t, err)
	require.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	require.NoError(t, err)
	backend.RegisterBackend("cuda", cu)

	cb := cpu.NewCPUBackend()

	batchSize, inChannels, inHeight, inWidth := 2, 3, 7, 6
	outChannels, kernelHeight, kernelWidth := 4, 3, 3
	padHeight, padWidth := 1, 1
	strideHeight, strideWidth := 1, 1

	inSize := batchSize * inChannels * inHeight * inWidth
	wSize := outChannels * inChannels * kernelHeight * kernelWidth
	bSize := outChannels

	hInput := make([]float32, inSize)
	hWeights := make([]float32, wSize)
	hBias := make([]float32, bSize)
	convRandomInit(hInput)
	convRandomInit(hWeights)
	convRandomInit(hBias)

	expected := cb.Conv2DForward(
		hInput, hWeights, hBias,
		batchSize, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		padHeight, padWidth,
		strideHeight, strideWidth,
	)

	dInput := cu.ToDevice(hInput)
	dWeights := cu.ToDevice(hWeights)
	dBias := cu.ToDevice(hBias)
	dOut := cu.Conv2DForward(
		dInput, dWeights, dBias,
		batchSize, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		padHeight, padWidth,
		strideHeight, strideWidth,
	)
	cu.Sync()

	actual := cu.ToCPU(dOut)
	require.Equal(t, len(expected), len(actual))
	assertSliceInDelta(t, expected, actual, 1e-3)

	cu.Free(dInput)
	cu.Free(dWeights)
	cu.Free(dBias)
	cu.Free(dOut)
}

func TestCudaConv2DBackwardMatchesCPU(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	require.NoError(t, err)
	require.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	require.NoError(t, err)
	backend.RegisterBackend("cuda", cu)

	cb := cpu.NewCPUBackend()

	batchSize, inChannels, inHeight, inWidth := 2, 2, 6, 5
	outChannels, kernelHeight, kernelWidth := 3, 3, 3
	padHeight, padWidth := 1, 1
	strideHeight, strideWidth := 1, 1

	outHeight := (inHeight+2*padHeight-kernelHeight)/strideHeight + 1
	outWidth := (inWidth+2*padWidth-kernelWidth)/strideWidth + 1

	inSize := batchSize * inChannels * inHeight * inWidth
	wSize := outChannels * inChannels * kernelHeight * kernelWidth
	outGradSize := batchSize * outChannels * outHeight * outWidth

	hInput := make([]float32, inSize)
	hWeights := make([]float32, wSize)
	hOutGrad := make([]float32, outGradSize)
	convRandomInit(hInput)
	convRandomInit(hWeights)
	convRandomInit(hOutGrad)

	expInGrad, expWGrad, expBGrad := cb.Conv2DBackward(
		hInput, hWeights, hOutGrad,
		batchSize, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		padHeight, padWidth,
		strideHeight, strideWidth,
	)

	dInput := cu.ToDevice(hInput)
	dWeights := cu.ToDevice(hWeights)
	dOutGrad := cu.ToDevice(hOutGrad)
	actInGradD, actWGradD, actBGradD := cu.Conv2DBackward(
		dInput, dWeights, dOutGrad,
		batchSize, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		padHeight, padWidth,
		strideHeight, strideWidth,
	)
	cu.Sync()

	actInGrad := cu.ToCPU(actInGradD)
	actWGrad := cu.ToCPU(actWGradD)
	actBGrad := cu.ToCPU(actBGradD)

	assertSliceInDelta(t, expInGrad, actInGrad, 1e-3)
	assertSliceInDelta(t, expWGrad, actWGrad, 1e-3)
	assertSliceInDelta(t, expBGrad, actBGrad, 1e-3)

	cu.Free(dInput)
	cu.Free(dWeights)
	cu.Free(dOutGrad)
	cu.Free(actInGradD)
	cu.Free(actWGradD)
	cu.Free(actBGradD)
}

func BenchmarkCudaConv2DForward(b *testing.B) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}

	backend.RegisterBackend("cuda", cu)

	batchSize, inChannels, inHeight, inWidth := 32, 16, 32, 32
	outChannels, kernelHeight, kernelWidth := 32, 3, 3
	padHeight, padWidth := 1, 1
	strideHeight, strideWidth := 1, 1

	inSize := batchSize * inChannels * inHeight * inWidth
	wSize := outChannels * inChannels * kernelHeight * kernelWidth
	bSize := outChannels

	hInput := make([]float32, inSize)
	hWeights := make([]float32, wSize)
	hBias := make([]float32, bSize)
	convRandomInit(hInput)
	convRandomInit(hWeights)
	convRandomInit(hBias)

	dInput := cu.ToDevice(hInput)
	dWeights := cu.ToDevice(hWeights)
	dBias := cu.ToDevice(hBias)

	for i := 0; i < 10; i++ {
		dOut := cu.Conv2DForward(
			dInput, dWeights, dBias,
			batchSize, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			padHeight, padWidth,
			strideHeight, strideWidth,
		)
		cu.Free(dOut)
	}
	cu.Sync()

	b.ResetTimer()
	b.SetBytes(int64((inSize + wSize) * 4))

	for i := 0; i < b.N; i++ {
		dOut := cu.Conv2DForward(
			dInput, dWeights, dBias,
			batchSize, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			padHeight, padWidth,
			strideHeight, strideWidth,
		)
		if i%10 == 0 {
			cu.Sync()
		}
		cu.Free(dOut)
	}
	cu.Sync()

	b.StopTimer()
	cu.Free(dInput)
	cu.Free(dWeights)
	cu.Free(dBias)
}

func BenchmarkCudaConv2DBackward(b *testing.B) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}

	backend.RegisterBackend("cuda", cu)

	batchSize, inChannels, inHeight, inWidth := 32, 16, 32, 32
	outChannels, kernelHeight, kernelWidth := 32, 3, 3
	padHeight, padWidth := 1, 1
	strideHeight, strideWidth := 1, 1

	outHeight := (inHeight+2*padHeight-kernelHeight)/strideHeight + 1
	outWidth := (inWidth+2*padWidth-kernelWidth)/strideWidth + 1

	inSize := batchSize * inChannels * inHeight * inWidth
	wSize := outChannels * inChannels * kernelHeight * kernelWidth
	outGradSize := batchSize * outChannels * outHeight * outWidth

	hInput := make([]float32, inSize)
	hWeights := make([]float32, wSize)
	hOutGrad := make([]float32, outGradSize)
	convRandomInit(hInput)
	convRandomInit(hWeights)
	convRandomInit(hOutGrad)

	dInput := cu.ToDevice(hInput)
	dWeights := cu.ToDevice(hWeights)
	dOutGrad := cu.ToDevice(hOutGrad)

	for i := 0; i < 5; i++ {
		dInGrad, dWGrad, dBGrad := cu.Conv2DBackward(
			dInput, dWeights, dOutGrad,
			batchSize, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			padHeight, padWidth,
			strideHeight, strideWidth,
		)
		cu.Free(dInGrad)
		cu.Free(dWGrad)
		cu.Free(dBGrad)
	}
	cu.Sync()

	b.ResetTimer()
	b.SetBytes(int64((inSize + wSize + outGradSize) * 4))

	for i := 0; i < b.N; i++ {
		dInGrad, dWGrad, dBGrad := cu.Conv2DBackward(
			dInput, dWeights, dOutGrad,
			batchSize, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			padHeight, padWidth,
			strideHeight, strideWidth,
		)
		if i%5 == 0 {
			cu.Sync()
		}
		cu.Free(dInGrad)
		cu.Free(dWGrad)
		cu.Free(dBGrad)
	}
	cu.Sync()

	b.StopTimer()
	cu.Free(dInput)
	cu.Free(dWeights)
	cu.Free(dOutGrad)
}

func convRandomInit(buf []float32) {
	for i := range buf {
		buf[i] = rand.Float32()
	}
}

func assertSliceInDelta(t *testing.T, expected, actual []float32, delta float64) {
	t.Helper()
	require.Equal(t, len(expected), len(actual))
	for i := range expected {
		assert.InDelta(t, expected[i], actual[i], delta, "Mismatch at index %d", i)
	}
}
