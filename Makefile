# nanograd -- development tasks
#
# The CUDA backend is behind the `cuda` build tag. Without it a stub is compiled
# and every CUDA constructor returns ErrCUDAUnavailable, so the library builds
# anywhere -- including with CGO_ENABLED=0.

GO      ?= go
PKGS    ?= ./...
CPU_PKGS = ./tensor/... ./storage/... ./optim/... ./nn/... ./internal/backend/cpu/... ./gradcheck/...

# Override if your CUDA toolkit lives elsewhere:
#   make test-cuda CUDA_PATH=/opt/cuda
CUDA_PATH ?= /usr/local/cuda

.PHONY: all
all: fmt vet build test

.PHONY: build
build:
	$(GO) build $(PKGS)

.PHONY: build-portable
build-portable: ## prove the library builds with no cgo toolchain at all
	CGO_ENABLED=0 $(GO) build $(PKGS)
	CGO_ENABLED=0 $(GO) test -run=NONE -count=1 $(PKGS)

.PHONY: test
test:
	$(GO) test -count=1 $(CPU_PKGS)

.PHONY: test-race
test-race:
	$(GO) test -race -count=1 $(CPU_PKGS)

.PHONY: gradcheck
gradcheck: ## verify every backward pass against finite differences
	$(GO) test -run Gradcheck -count=1 -v ./tensor/

.PHONY: cover
cover:
	$(GO) test -count=1 -coverprofile=coverage.out $(CPU_PKGS)
	$(GO) tool cover -func=coverage.out | tail -1

.PHONY: cover-html
cover-html: cover
	$(GO) tool cover -html=coverage.out

.PHONY: canary
canary: ## MNIST MLP, >95% gate, ~10s -- the only whole-stack check
	$(GO) test -run 'TestMNIST$$' -v -count=1 -timeout 20m ./example/MNIST/

.PHONY: canary-all
canary-all: ## adds the CNN (~100s) and CIFAR-10
	$(GO) test -run 'TestMNIST$$|TestMNISTCNN' -v -count=1 -timeout 60m ./example/MNIST/
	$(GO) test -run TestCIFAR10_CNN -v -count=1 -timeout 90m ./example/CIFAR-10/

.PHONY: kernels
kernels: ## compile the CUDA kernels (requires nvcc)
	$(MAKE) -C internal/backend/cuda/kernels clean
	$(MAKE) -C internal/backend/cuda/kernels all -j8

.PHONY: build-cuda
build-cuda: ## build with the CUDA backend enabled
	CGO_ENABLED=1 CGO_CFLAGS="-I$(CUDA_PATH)/include" \
		CGO_LDFLAGS="-L$(CUDA_PATH)/lib64" \
		$(GO) build -tags cuda $(PKGS)

.PHONY: test-cuda
test-cuda: ## run the CUDA test suite (requires a GPU)
	CGO_ENABLED=1 CGO_CFLAGS="-I$(CUDA_PATH)/include" \
		CGO_LDFLAGS="-L$(CUDA_PATH)/lib64" \
		$(GO) test -tags cuda -count=1 -v ./internal/backend/cuda/...

.PHONY: fmt
fmt:
	gofmt -w .

.PHONY: fmt-check
fmt-check:
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "not gofmt'd:"; echo "$$unformatted"; exit 1; \
	fi

.PHONY: vet
vet:
	$(GO) vet $(PKGS)

.PHONY: help
help:
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
