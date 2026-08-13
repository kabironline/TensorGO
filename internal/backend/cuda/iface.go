//go:build cuda

package cuda

import "github.com/kabironline/nanograd/backend"

// Compile-time proof that CUDABackend satisfies the full Backend contract.
// See the CPU counterpart for why this is worth having.
var _ backend.Backend = (*CUDABackend)(nil)
