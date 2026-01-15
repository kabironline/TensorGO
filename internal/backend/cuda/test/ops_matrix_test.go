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

	size := 1024
	a := tensor.NewIdentityTensor(size)
	b := tensor.NewIdentityTensor(size)

	c := a.MatMul(b)
	assert.NotNil(t, c)

	cu.Sync()

	// Validate on CPU to avoid huge managed-memory page migrations.
	cHost := cu.ToCPU(c.Data)
	// spot-check diagonal
	step := size / 8
	if step == 0 {
		step = 1
	}
	for i := 0; i < size; i += step {
		idx := i*size + i
		assert.Equal(t, float32(1.0), cHost[idx])
	}
	// spot-check a few off-diagonal entries
	if size > 1 {
		assert.Equal(t, float32(0.0), cHost[1])
		assert.Equal(t, float32(0.0), cHost[size])
		last := size*size - 1
		assert.Equal(t, float32(0.0), cHost[last-1])
	}
}

func BenchmarkCudaMatMul(b *testing.B) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}

	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)

	size := 4096
	h_a := make([]float32, size*size)
	h_b := make([]float32, size*size)

	d_a := cu.ToDevice(h_a)
	d_b := cu.ToDevice(h_b)
	d_c := cu.Allocate(size * size)

	// Warmup
	for i := 0; i < 10; i++ {
		cu.MatMul(
			d_a,
			d_b,
			d_c,
			size, size, size,
			size, size,
		)
	}
	cu.Sync()

	b.ResetTimer()
	b.SetBytes(int64(size * size * 4 * 3))

	// Benchmark (sync periodically to avoid unbounded queue growth)
	for i := 0; i < b.N; i++ {
		cu.MatMul(
			d_a,
			d_b,
			d_c,
			size, size, size,
			size, size,
		)
		if i%10 == 0 {
			cu.Sync()
		}
	}
	cu.Sync()

	b.StopTimer()
	cu.Free(d_a)
	cu.Free(d_b)
	cu.Free(d_c)
}
