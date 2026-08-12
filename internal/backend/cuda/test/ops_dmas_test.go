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

	res := cu.ToCPU(c.Data())
	for i := 0; i < size; i++ {
		assert.Equal(t, hA[i]+hB[i], res[i])
	}

	// Sub test
	c = a.Sub(b)
	assert.NotNil(t, c)
	cu.Sync()

	res = cu.ToCPU(c.Data())
	for i := 0; i < size; i++ {
		assert.Equal(t, hA[i]-hB[i], res[i])
	}

	// Mul test
	c = a.Mul(b)
	assert.NotNil(t, c)
	cu.Sync()

	res = cu.ToCPU(c.Data())
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

// TestCudaAddVectorized tests the vectorized add kernel (using float4)
// This validates both the vectorized path (for sizes divisible by 4)
// and the scalar remainder path (for non-aligned sizes).
func TestCudaAddVectorized(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	backend.RegisterBackend("cuda", cu)

	// Test various sizes to verify both float4 vectorized path and scalar remainder
	testCases := []int{
		1,    // Single element
		3,    // Not aligned to 4
		4,    // Exactly one float4
		7,    // Vectorized + remainder
		16,   // Multiple float4s, no remainder
		100,  // Mixed
		1024, // Larger aligned
		1025, // Larger with remainder
	}

	for _, size := range testCases {
		t.Run(string(rune('0'+size/100))+string(rune('0'+(size/10)%10))+string(rune('0'+size%10)), func(t *testing.T) {
			a_data := make([]float32, size)
			b_data := make([]float32, size)

			for i := 0; i < size; i++ {
				a_data[i] = float32(i) * 0.5
				b_data[i] = float32(i) * 1.5
			}

			a := tensor.NewTensor(a_data, []int{size})
			b := tensor.NewTensor(b_data, []int{size})

			c := a.Add(b)
			assert.NotNil(t, c)
			cu.Sync()

			result := cu.ToCPU(c.Data())

			for i := 0; i < size; i++ {
				expected := a_data[i] + b_data[i]
				assert.InDelta(t, expected, result[i], 0.001, "Mismatch at index %d for size %d", i, size)
			}
		})
	}
}

func TestCudaSubVectorized(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	backend.RegisterBackend("cuda", cu)

	// Test various sizes including non-multiples of 4
	testCases := []int{1, 3, 4, 7, 16, 100, 1024, 1025}

	for _, size := range testCases {
		t.Run(string(rune('0'+size/100))+string(rune('0'+(size/10)%10))+string(rune('0'+size%10)), func(t *testing.T) {
			a_data := make([]float32, size)
			b_data := make([]float32, size)

			for i := 0; i < size; i++ {
				a_data[i] = float32(i) * 2.0
				b_data[i] = float32(i) * 0.5
			}

			a := tensor.NewTensor(a_data, []int{size})
			b := tensor.NewTensor(b_data, []int{size})

			c := a.Sub(b)
			assert.NotNil(t, c)
			cu.Sync()

			result := cu.ToCPU(c.Data())

			for i := 0; i < size; i++ {
				expected := a_data[i] - b_data[i]
				assert.InDelta(t, expected, result[i], 0.001, "Mismatch at index %d for size %d", i, size)
			}
		})
	}
}

func TestCudaAddSubEdgeCases(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	// Test with zero inputs
	t.Run("ZeroInputs", func(t *testing.T) {
		size := 100
		a_data := make([]float32, size)
		b_data := make([]float32, size)
		// Leave as zeros

		a := tensor.NewTensor(a_data, []int{size})
		b := tensor.NewTensor(b_data, []int{size})

		c := a.Add(b)
		assert.NotNil(t, c)
		cu.Sync()

		result := cu.ToCPU(c.Data())
		for i := 0; i < size; i++ {
			assert.Equal(t, float32(0), result[i])
		}
	})

	// Test with negative values
	t.Run("NegativeValues", func(t *testing.T) {
		size := 50
		a_data := make([]float32, size)
		b_data := make([]float32, size)

		for i := 0; i < size; i++ {
			a_data[i] = float32(-i)
			b_data[i] = float32(i)
		}

		a := tensor.NewTensor(a_data, []int{size})
		b := tensor.NewTensor(b_data, []int{size})

		c := a.Add(b)
		assert.NotNil(t, c)
		cu.Sync()

		result := cu.ToCPU(c.Data())
		for i := 0; i < size; i++ {
			assert.Equal(t, float32(0), result[i])
		}
	})

	// Test subtraction resulting in negative
	t.Run("SubtractionNegativeResult", func(t *testing.T) {
		size := 20
		a_data := make([]float32, size)
		b_data := make([]float32, size)

		for i := 0; i < size; i++ {
			a_data[i] = 1.0
			b_data[i] = 2.0
		}

		a := tensor.NewTensor(a_data, []int{size})
		b := tensor.NewTensor(b_data, []int{size})

		c := a.Sub(b)
		assert.NotNil(t, c)
		cu.Sync()

		result := cu.ToCPU(c.Data())
		for i := 0; i < size; i++ {
			assert.Equal(t, float32(-1.0), result[i])
		}
	})
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
