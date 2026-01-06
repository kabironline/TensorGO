//go:build cuda

package cuda_test

import (
	"testing"

	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/stretchr/testify/assert"
)

func TestCUDABackendInitialization(t *testing.T) {
	_, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err, "Failed to initialize CUDA backend")
}

func TestCUDADeviceSelection(t *testing.T) {
	b, err := cuda.NewCUDABackend(0)
	assert.NoError(t, err, "Failed to select CUDA device 0")
	assert.Equal(t, "cuda:0", b.Name(), "Backend name should be 'cuda:0'")
}

func TestCUDADeviceProperties(t *testing.T) {
	b, err := cuda.GetCudaDeviceCount()
	assert.NoError(t, err, "Failed to get CUDA device count")
	assert.Greater(t, b, 0, "There should be at least one CUDA device available")

	// Printing device properties for debugging
	for i := 0; i < b; i++ {
		props, err := cuda.GetCudaDeviceProps(i)
		assert.NoError(t, err, "Failed to get properties for device %d", i)
		t.Logf("Device %d: %s, Compute Capability: %d.%d, Total Memory: %d MB",
			i, props["Name"], props["Major"], props["Minor"], props["TotalGlobalMem"].(uint64)/(1024*1024))
	}
}
