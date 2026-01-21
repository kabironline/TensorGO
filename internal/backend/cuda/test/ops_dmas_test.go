package cuda_test

import (
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/kabironline/nanograd/tensor"
	"github.com/stretchr/testify/assert"
)

func TestCudaDMAS(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	// Register and set default so helper constructors can find it
	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)

	size := 1024
	hA := make([]float32, size)
	hB := make([]float32, size)
	for i := 0; i < size; i++ {
		hA[i] = float32(i)
		hB[i] = 1.5
	}

	a := tensor.NewTensor(hA, []int{size})
	b := tensor.NewTensor(hB, []int{size})

	// Add test
	c := a.Add(b)
	assert.NotNil(t, c)
	cu.Sync()

	res := cu.ToCPU(c.Data)
	for i := 0; i < size; i++ {
		assert.Equal(t, hA[i]+hB[i], res[i])
	}

	// Sub test
	c = a.Sub(b)
	assert.NotNil(t, c)
	cu.Sync()

	res = cu.ToCPU(c.Data)
	for i := 0; i < size; i++ {
		assert.Equal(t, hA[i]-hB[i], res[i])
	}

	// Mul test
	c = a.Mul(b)
	assert.NotNil(t, c)
	cu.Sync()

	res = cu.ToCPU(c.Data)
	for i := 0; i < size; i++ {
		assert.Equal(t, hA[i]*hB[i], res[i])
	}
}

func BenchmarkCudaAdd(b *testing.B) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}

	size := 1 << 20 // 1M elements
	hA := make([]float32, size)
	hB := make([]float32, size)
	for i := 0; i < size; i++ {
		hA[i] = 1.0
		hB[i] = 2.0
	}
	dA := cu.ToDevice(hA)
	dB := cu.ToDevice(hB)
	dOut := cu.Allocate(size)

	// Warmup
	for i := 0; i < 10; i++ {
		cu.Add(dA, dB, dOut, size)
	}
	cu.Sync()

	b.ResetTimer()
	b.SetBytes(int64(size * 4 * 3))

	for i := 0; i < b.N; i++ {
		cu.Add(dA, dB, dOut, size)
		if i%10 == 0 {
			cu.Sync()
		}
	}
	cu.Sync()
}

func BenchmarkCudaMul(b *testing.B) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}

	// Million element vectors
	size := 1 << 20
	hA := make([]float32, size)
	hB := make([]float32, size)
	for i := 0; i < size; i++ {
		hA[i] = 1.0
		hB[i] = 2.0
	}
	dA := cu.ToDevice(hA)
	dB := cu.ToDevice(hB)
	dOut := cu.Allocate(size)

	// Warmup
	for i := 0; i < 10; i++ {
		cu.Mul(dA, dB, dOut, size)
	}
	cu.Sync()

	b.ResetTimer()
	b.SetBytes(int64(size * 4 * 3))

	for i := 0; i < b.N; i++ {
		cu.Mul(dA, dB, dOut, size)
		if i%10 == 0 {
			cu.Sync()
		}
	}
	cu.Sync()
}
