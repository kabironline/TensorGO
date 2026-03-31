package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lcudart -lcublas -lcudnn ${SRCDIR}/kernels/libcuda.a -lm
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR} -I${SRCDIR}/kernels
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/kabironline/nanograd/backend"
)

type CUDABackend struct {
	backend.Base

	deviceID int
	// Handles should be managed as pointers to C types if using CGO
	// or kept as unsafe.Pointer if using lazy loading (libloading).
	cuBLASHandle unsafe.Pointer
	cuDNNHandle  unsafe.Pointer
	stream       unsafe.Pointer

	memPool *MemoryPool

	workspace     unsafe.Pointer
	workspaceSize int

	// prevMath stores the previous cuBLAS math mode (so we can restore it on cleanup)
	prevMath C.cublasMath_t
	// whether we successfully set the tensor-op math mode during init
	mathModeChanged bool
}

func NewCUDABackend(deviceID int) (*CUDABackend, error) {
	// 1. Switch context to the target GPU
	res := C.cudaSetDevice(C.int(deviceID))
	if res != C.cudaSuccess {
		return nil, fmt.Errorf("failed to set cuda device %d", deviceID)
	}

	b := &CUDABackend{
		Base:     *backend.NewBase(fmt.Sprintf("cuda:%d", deviceID), true),
		deviceID: deviceID,
		memPool:  NewMemoryPool(0), // Use the Power-of-2 pool we discussed
	}

	// 2. Initialize cuBLAS
	var cublasHandle C.cublasHandle_t
	if C.cublasCreate(&cublasHandle) != C.CUBLAS_STATUS_SUCCESS {
		return nil, fmt.Errorf("failed to initialize cuBLAS")
	}
	b.cuBLASHandle = unsafe.Pointer(cublasHandle)

	// 3. Initialize cuDNN
	var cudnnHandle C.cudnnHandle_t
	if C.cudnnCreate(&cudnnHandle) != C.CUDNN_STATUS_SUCCESS {
		return nil, fmt.Errorf("failed to initialize cuDNN")
	}
	b.cuDNNHandle = unsafe.Pointer(cudnnHandle)

	// 4. Create CUDA Stream
	var stream C.cudaStream_t
	if C.cudaStreamCreate(&stream) != C.cudaSuccess {
		return nil, fmt.Errorf("failed to create cuda stream")
	}
	b.stream = unsafe.Pointer(stream)

	// bind stream to cuBLAS
	if C.cublasSetStream(C.cublasHandle_t(cublasHandle), stream) != C.CUBLAS_STATUS_SUCCESS {
		return nil, fmt.Errorf("failed to set cuBLAS stream")
	}

	// Best-effort: attempt to enable Tensor Core / TF32 math mode for faster GEMMs
	// Record previous mode so we can restore it in the finalizer.
	var prev C.cublasMath_t
	if C.cublasGetMathMode(C.cublasHandle_t(cublasHandle), &prev) == C.CUBLAS_STATUS_SUCCESS {
		if C.cublasSetMathMode(C.cublasHandle_t(cublasHandle), C.CUBLAS_TENSOR_OP_MATH) == C.CUBLAS_STATUS_SUCCESS {
			b.prevMath = prev
			b.mathModeChanged = true
		}
	}

	// 5. Set up cleanup
	runtime.SetFinalizer(b, func(obj *CUDABackend) {
		// Restore previous cuBLAS math mode if we changed it during init
		if obj.mathModeChanged {
			C.cublasSetMathMode(C.cublasHandle_t(obj.cuBLASHandle), obj.prevMath)
		}
		C.cublasDestroy(C.cublasHandle_t(obj.cuBLASHandle))
		// C.cudnnDestroy(C.cudnnHandle_t(obj.cuDNNHandle))
		if obj.memPool != nil {
			obj.memPool.Clear()
		}
		if obj.stream != nil {
			C.cudaStreamDestroy(C.cudaStream_t(obj.stream))
		}
	})

	return b, nil
}

func GetCudaDeviceCount() (int, error) {
	var count C.int
	err := C.cudaGetDeviceCount(&count)
	if err != C.cudaSuccess {
		return 0, fmt.Errorf("failed to get cuda device count")
	}
	return int(count), nil
}

func GetCudaDeviceProps(deviceID int) (map[string]interface{}, error) {
	var props C.struct_cudaDeviceProp
	res := C.cudaGetDeviceProperties(&props, C.int(deviceID))
	if res != C.cudaSuccess {
		// Use C.cudaGetErrorString for better debugging info
		errStr := C.GoString(C.cudaGetErrorString(res))
		return nil, fmt.Errorf("cuda error: %s (device %d)", errStr, deviceID)
	}

	info := make(map[string]interface{})
	info["Name"] = C.GoString((*C.char)(&props.name[0]))

	info["TotalGlobalMem"] = uint64(props.totalGlobalMem)
	info["SharedMemPerBlock"] = uint64(props.sharedMemPerBlock)
	info["RegsPerBlock"] = int(props.regsPerBlock)
	info["WarpSize"] = int(props.warpSize)
	info["MemPitch"] = uint64(props.memPitch)
	info["MaxThreadsPerBlock"] = int(props.maxThreadsPerBlock)

	// Arrays must be accessed by index as you correctly did
	info["MaxThreadsDim"] = []int{int(props.maxThreadsDim[0]), int(props.maxThreadsDim[1]), int(props.maxThreadsDim[2])}
	info["MaxGridSize"] = []int{int(props.maxGridSize[0]), int(props.maxGridSize[1]), int(props.maxGridSize[2])}

	// NEW (Compatible with CUDA 12 and 13)
	var clockRateKHz C.int
	C.cudaDeviceGetAttribute(&clockRateKHz, C.cudaDevAttrClockRate, C.int(deviceID))

	info["TotalConstMem"] = uint64(props.totalConstMem)
	info["Major"] = int(props.major)
	info["Minor"] = int(props.minor)
	info["ClockRate"] = int(clockRateKHz)
	info["MultiProcessorCount"] = int(props.multiProcessorCount)

	return info, nil
}
