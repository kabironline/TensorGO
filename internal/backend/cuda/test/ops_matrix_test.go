package cuda_test

import (
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/kabironline/nanograd/tensor"
	"github.com/stretchr/testify/assert"
)

func TestCudaMatMul(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	// Register and set default so helper constructors like NewIdentityTensor can find it
	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)

	size := 4096
	a := tensor.NewIdentityTensor(size)
	b := tensor.NewIdentityTensor(size)

	c := a.MatMul(b)
	assert.NotNil(t, c)

	cu.Sync()

	// checking data of c
	// since c is a matrix multiplication of two identity matrices, the result should be an identity matrix, its data should be the same as the input matrices
	aData := a.Data
	bData := b.Data
	cData := c.Data

	assert.Equal(t, aData, cData)
	assert.Equal(t, bData, cData)
}

func BenchmarkCudaMatMul(b *testing.B) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}
	// Register and set default so helper constructors like NewIdentityTensor can find it
	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)

	size := 4096
	a := tensor.NewIdentityTensor(size)
	bb := tensor.NewIdentityTensor(size)

	// Warmup
	for i := 0; i < 10; i++ {
		a.MatMul(bb)
	}
	cu.Sync()

	b.ResetTimer()

	// Benchmark
	for b.Loop() {
		a.MatMul(bb)
	}
	cu.Sync()
}
