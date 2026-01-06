//go:build !cuda

package cuda

import (
	"fmt"

	"github.com/kabironline/nanograd/backend"
)

// CUDABackend is a dummy struct when CUDA is not enabled at build time.
type CUDABackend struct {
	backend.Base
}

// NewCUDABackend returns an error since CUDA support was not compiled in.
func NewCUDABackend(deviceID int) (*CUDABackend, error) {
	return nil, fmt.Errorf("nanograd was built without CUDA support. Use '-tags cuda' during build")
}

// MemoryPool is a dummy struct.
type MemoryPool struct{}

// WorkspaceManager is a dummy struct.
type WorkspaceManager struct{}

// No init function here means the "cuda" backend won't be registered
// in the global backend map, allowing it to fall back to CPU.
