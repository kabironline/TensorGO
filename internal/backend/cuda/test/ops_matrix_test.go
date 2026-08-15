//go:build cuda

package cuda_test

import (
	"math/rand/v2"
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
	cHost := cu.ToCPU(c.Data())
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

	m := 4096
	k := 2048
	n := 1024

	h_a := make([]float32, m*k)
	h_b := make([]float32, k*n)

	randomInit(h_a)
	randomInit(h_b)

	d_a := cu.ToDevice(h_a)
	d_b := cu.ToDevice(h_b)
	d_c := cu.Allocate(m * n * 4)

	// Warmup
	for i := 0; i < 10; i++ {
		cu.MatMul(
			backend.MatOperand{Data: d_a, Rows: m, Cols: k, LD: k},
			backend.MatOperand{Data: d_b, Rows: k, Cols: n, LD: n},
			d_c,
			1.0, 0.0,
		)
	}
	cu.Sync()

	b.ResetTimer()
	b.SetBytes(int64(m * n * 4 * 3))

	// Benchmark (sync periodically to avoid unbounded queue growth)
	for i := 0; i < b.N; i++ {
		cu.MatMul(
			backend.MatOperand{Data: d_a, Rows: m, Cols: k, LD: k},
			backend.MatOperand{Data: d_b, Rows: k, Cols: n, LD: n},
			d_c,
			1.0, 0.0,
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

func TestCudaMatMulTransA(t *testing.T) {
	// Tests the fixed trans_a formula: sgemm(OP_N, OP_T, n, m, k, B, n, A, m, C, n)
	// The critical fix was using lda=n instead of lda=m for matrix B
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	backend.RegisterBackend("cuda", cu)

	// Test case: C[2×2] = A^T @ B where A is [3×2], B is [3×2]
	// A = [[1,2], [3,4], [5,6]]
	// A^T = [[1,3,5], [2,4,6]]
	// B = [[7,8], [9,10], [11,12]]
	// Expected C = [[1*7+3*9+5*11, 1*8+3*10+5*12],
	//               [2*7+4*9+6*11, 2*8+4*10+6*12]]
	//            = [[89, 98], [116, 128]]

	a_data := []float32{1, 2, 3, 4, 5, 6}
	b_data := []float32{7, 8, 9, 10, 11, 12}

	a := tensor.NewTensor(a_data, []int{3, 2})
	b := tensor.NewTensor(b_data, []int{3, 2})

	// Manually call MatMulTransA through the backend
	m, k, n := 2, 3, 2 // Result is [2×2], A^T is [2×3], B is [3×2]
	result := cu.Allocate(m * n)
	cu.MatMul(
		backend.MatOperand{Data: a.Data(), Rows: k, Cols: m, LD: m}.T(),
		backend.MatOperand{Data: b.Data(), Rows: k, Cols: n, LD: n},
		result, 1.0, 0.0,
	)
	cu.Sync()

	result_cpu := cu.ToCPU(result)

	// Verify the result
	expected := []float32{89, 98, 116, 128}
	for i, exp := range expected {
		assert.InDelta(t, exp, result_cpu[i], 0.01, "Mismatch at index %d", i)
	}
}

func TestCudaMatMulTransB(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	backend.RegisterBackend("cuda", cu)

	// Test case: C[2×2] = A @ B^T where A is [2×3], B is [2×3]
	// A = [[1,2,3], [4,5,6]]
	// B = [[7,9,11], [8,10,12]]
	// B^T = [[7,8], [9,10], [11,12]]
	// Expected C = [[1*7+2*9+3*11, 1*8+2*10+3*12],
	//               [4*7+5*9+6*11, 4*8+5*10+6*12]]
	//            = [[58, 64], [139, 154]]

	a_data := []float32{1, 2, 3, 4, 5, 6}
	b_data := []float32{7, 9, 11, 8, 10, 12}

	a := tensor.NewTensor(a_data, []int{2, 3})
	b := tensor.NewTensor(b_data, []int{2, 3})

	// Manually call MatMulTransB through the backend
	m, k, n := 2, 3, 2 // Result is [2×2], A is [2×3], B^T is [3×2]
	result := cu.Allocate(m * n)
	cu.MatMul(
		backend.MatOperand{Data: a.Data(), Rows: m, Cols: k, LD: k},
		backend.MatOperand{Data: b.Data(), Rows: n, Cols: k, LD: k}.T(),
		result, 1.0, 0.0,
	)
	cu.Sync()

	result_cpu := cu.ToCPU(result)

	// Verify the result
	expected := []float32{58, 64, 139, 154}
	for i, exp := range expected {
		assert.InDelta(t, exp, result_cpu[i], 0.01, "Mismatch at index %d", i)
	}
}

func TestCudaMatMulTransA_NonSquare(t *testing.T) {
	// CRITICAL TEST: This test with m≠n would have immediately caught the lda bug.
	// When m=n=2 (square result), both lda=m and lda=n give the same value,
	// hiding the bug. With m=3, n=4, the bug becomes obvious.
	// This demonstrates why testing with non-square matrices is essential.
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	backend.RegisterBackend("cuda", cu)

	// Test with non-square result: C[3×4] = A^T @ B where A is [5×3], B is [5×4]
	// This test case specifically validates the leading dimension fix (lda=n not lda=m)
	m, k, n := 3, 5, 4

	a_data := make([]float32, k*m) // [5×3]
	b_data := make([]float32, k*n) // [5×4]

	// Initialize with simple pattern for verification
	for i := 0; i < k*m; i++ {
		a_data[i] = float32(i + 1)
	}
	for i := 0; i < k*n; i++ {
		b_data[i] = float32(i + 1)
	}

	a := tensor.NewTensor(a_data, []int{k, m})
	b := tensor.NewTensor(b_data, []int{k, n})

	result := cu.Allocate(m * n)
	cu.MatMul(
		backend.MatOperand{Data: a.Data(), Rows: k, Cols: m, LD: m}.T(),
		backend.MatOperand{Data: b.Data(), Rows: k, Cols: n, LD: n},
		result, 1.0, 0.0,
	)
	cu.Sync()

	result_cpu := cu.ToCPU(result)

	// Compute expected result on CPU for comparison
	expected := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for kk := 0; kk < k; kk++ {
				// A^T[i,kk] = A[kk,i]
				// B[kk,j]
				a_val := a_data[kk*m+i]
				b_val := b_data[kk*n+j]
				sum += a_val * b_val
			}
			expected[i*n+j] = sum
		}
	}

	// Verify
	for i := 0; i < m*n; i++ {
		assert.InDelta(t, expected[i], result_cpu[i], 0.01, "Mismatch at index %d", i)
	}
}

func TestCudaMatMulTransB_LargerMatrix(t *testing.T) {
	devices, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, devices, 1)

	cu, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	backend.RegisterBackend("cuda", cu)

	// Larger test: ensure it works with typical neural network dimensions
	m, k, n := 32, 128, 64 // batch_size=32, features=128, hidden=64

	a_data := make([]float32, m*k)
	b_data := make([]float32, n*k)
	randomInit(a_data)
	randomInit(b_data)

	a := tensor.NewTensor(a_data, []int{m, k})
	b := tensor.NewTensor(b_data, []int{n, k})

	result := cu.Allocate(m * n)
	cu.MatMul(
		backend.MatOperand{Data: a.Data(), Rows: m, Cols: k, LD: k},
		backend.MatOperand{Data: b.Data(), Rows: n, Cols: k, LD: k}.T(),
		result, 1.0, 0.0,
	)
	cu.Sync()

	result_cpu := cu.ToCPU(result)

	// Compute expected result on CPU
	expected := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for kk := 0; kk < k; kk++ {
				// A[i,kk] @ B^T[kk,j] = A[i,kk] @ B[j,kk]
				a_val := a_data[i*k+kk]
				b_val := b_data[j*k+kk]
				sum += a_val * b_val
			}
			expected[i*n+j] = sum
		}
	}

	// Verify (use larger tolerance for accumulated floating point errors)
	for i := 0; i < m*n; i++ {
		assert.InDelta(t, expected[i], result_cpu[i], 0.1, "Mismatch at index %d", i)
	}
}

func randomInit(buf []float32) {
	for i := range buf {
		buf[i] = rand.Float32()
	}
}
