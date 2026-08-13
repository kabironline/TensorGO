# nanograd

A small deep-learning library in Go, with an optional CUDA backend.

Define-by-run autograd, a pluggable backend interface, and enough neural-network
layers to train real models: MNIST reaches **95.5%** with an MLP and **98.7%**
with a CNN.

> **Status: pre-1.0, under active redesign.** The API will change. See
> [WIP-notes/Production-Readiness-Roadmap-2026-08-12.md](WIP-notes/Production-Readiness-Roadmap-2026-08-12.md)
> for what is known-broken and the order it is being fixed in. Notably, several
> operations on **non-contiguous tensors** (`Slice`, and `Add` on a transposed
> view) have known-incorrect gradients — they are marked with skipped gradient
> checks in `tensor/gradcheck_test.go`.

## Install

```bash
go get github.com/kabironline/nanograd
```

CUDA is **not** required. Without the `cuda` build tag a stub backend is compiled
and the library builds anywhere, including with `CGO_ENABLED=0`.

## Quick start

```go
package main

import (
	"fmt"

	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/nn/activations"
	"github.com/kabironline/nanograd/optim"
	"github.com/kabironline/nanograd/tensor"
)

func main() {
	model := nn.NewSequential(
		nn.NewLinear(2, 8),
		&activations.ReLU{},
		nn.NewLinear(8, 1),
	)
	opt := optim.NewAdam(model.Parameters(), 0.01)

	x := tensor.NewTensor([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y := tensor.NewTensor([]float32{0, 1, 1, 0}, []int{4, 1})

	for epoch := 0; epoch < 1000; epoch++ {
		opt.ZeroGrad()
		loss := nn.MSELoss(model.Forward(x), y)
		loss.BackProp()
		opt.Step()
	}
	fmt.Println(model.Forward(x).Data())
}
```

## Using the GPU

The CUDA backend is behind a build tag because it needs cgo and the CUDA toolkit:

```bash
make kernels            # compile the .cu kernels (needs nvcc)
go build -tags cuda ./...
go test  -tags cuda ./internal/backend/cuda/...
```

Without the tag, `cuda.NewCUDABackend` returns `cuda.ErrCUDAUnavailable` and
`cuda.GetCudaDeviceCount` reports zero devices, so GPU code paths skip cleanly.

If your toolkit is not in `/usr/local/cuda`:

```bash
make build-cuda CUDA_PATH=/opt/cuda
```

## Development

```bash
make            # fmt, vet, build, test
make gradcheck  # verify every backward pass against finite differences
make canary     # MNIST MLP, >95% gate, ~10s
make test-race  # race detector
make cover      # coverage summary
```

`make build-portable` is the one that matters before releasing: it proves the
library still compiles with `CGO_ENABLED=0`.

### Gradient checking

Hand-written backward passes are easy to get subtly wrong, so
[`gradcheck`](gradcheck/gradcheck.go) compares every analytic gradient against a
central finite difference:

```go
gradcheck.Check(t, "matmul/transposed-rhs", func() *tensor.Tensor {
	return x.MatMul(y.Transpose([]int{1, 0}))
}, x, y)
```

Inputs are contiguous leaves; views belong inside the closure, which is rebuilt
on every perturbed pass. The loss is a *position-weighted* sum rather than a
plain one — a plain sum is permutation-invariant, so an op that reads a strided
view in the wrong order would produce an identical loss and go undetected.

Add a case to `tensor/gradcheck_test.go` for every new op, and cross it with
non-contiguous inputs: that is where the bugs are.

## Layout

| Path | What it is |
|---|---|
| `tensor/` | `Tensor`, autograd engine, ops |
| `storage/` | dtype-tagged, device-agnostic buffers |
| `backend/` | the `Backend` interface and registry |
| `internal/backend/cpu/` | CPU backend (gonum BLAS, worker pool) |
| `internal/backend/cuda/` | CUDA backend (cgo, `cuda` tag) + `.cu` kernels |
| `nn/` | layers, activations, losses, `Sequential` |
| `optim/` | SGD, Adam |
| `gradcheck/` | finite-difference gradient verification |
| `example/` | MNIST and CIFAR-10 accuracy canaries |

Datasets are gitignored. The canaries **skip** when the data is absent rather
than failing; `.github/workflows/ci.yml` shows how to fetch MNIST.

## License

See [LICENSE](LICENSE).
