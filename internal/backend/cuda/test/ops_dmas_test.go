package cuda_test

import (
	"testing"

	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/stretchr/testify/assert"
)

func TestCudaDMAS(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	if err != nil || devices < 1 {
		t.Skipf("CUDA not available: %v", err)
	}

	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		t.Skipf("CUDA backend init failed: %v", err)
	}

	// Init
	size := 1024
	hA := make([]float32, size)
	hB := make([]float32, size)
	for i := 0; i < size; i++ {
		hA[i] = float32(i)
		hB[i] = float32(1.5)
	}

	// Add test
	dA := cu.ToDevice(hA)
	dB := cu.ToDevice(hB)
	dOut := cu.Allocate(size)

	cu.Add(dA, dB, dOut, size)
	cu.Sync()

	res := cu.ToCPU(dOut)
	for i := 0; i < size; i++ {
		assert.Equal(t, hA[i]+hB[i], res[i])
	}

	// Sub test
	cu.Sub(dA, dB, dOut, size)
	cu.Sync()

	res = cu.ToCPU(dOut)
	for i := 0; i < size; i++ {
		assert.Equal(t, hA[i]-hB[i], res[i])
	}

	// Mul test
	cu.Mul(dA, dB, dOut, size)
	cu.Sync()

	res = cu.ToCPU(dOut)
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
