//go:build cuda

package cuda_test

import (
	"testing"

	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/stretchr/testify/assert"
)

func TestCUDAMemoryTransfer(t *testing.T) {
	b, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err)

	// Create test data
	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0}

	// Move to GPU
	gpuData := b.ToDevice(input)
	assert.NotNil(t, gpuData)
	assert.Equal(t, len(input), len(gpuData))

	// Move back to CPU
	output := b.ToCPU(gpuData)
	assert.Equal(t, input, output, "Data transferred back from GPU should match original")

	// Cleanup
	b.Free(gpuData)
}
