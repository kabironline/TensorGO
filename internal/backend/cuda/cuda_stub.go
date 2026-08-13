//go:build !cuda

// Package cuda provides a CUDA-backed implementation of backend.Backend.
//
// The real implementation is cgo and requires the CUDA toolkit, so it is guarded
// by the `cuda` build tag. Without that tag this stub is compiled instead, which
// keeps `go build ./...` working on machines with no CUDA toolchain (and with
// CGO_ENABLED=0) at the cost of every constructor returning ErrCUDAUnavailable.
//
// Build with CUDA:
//
//	go build -tags cuda ./...
//	go test  -tags cuda ./internal/backend/cuda/...
package cuda

import "errors"

// ErrCUDAUnavailable is returned by every constructor in this package when the
// binary was built without the `cuda` build tag.
var ErrCUDAUnavailable = errors.New(
	"cuda: this binary was built without CUDA support; rebuild with -tags cuda")

// CUDABackend is a placeholder for the real cgo-backed type. It exists so that
// code referring to *cuda.CUDABackend still compiles without the tag; it has no
// methods and cannot be constructed successfully.
type CUDABackend struct{}

// NewCUDABackend always fails in a non-CUDA build.
func NewCUDABackend(deviceID int) (*CUDABackend, error) {
	return nil, ErrCUDAUnavailable
}

// GetCudaDeviceCount reports zero devices in a non-CUDA build.
//
// It deliberately returns a nil error so that callers can use the common
// "count == 0 means skip the GPU path" pattern without special-casing the build
// tag. Use NewCUDABackend if you need to distinguish "no GPU present" from
// "not compiled with CUDA support".
func GetCudaDeviceCount() (int, error) { return 0, nil }

// GetCudaDeviceProps always fails in a non-CUDA build.
func GetCudaDeviceProps(deviceID int) (map[string]interface{}, error) {
	return nil, ErrCUDAUnavailable
}
