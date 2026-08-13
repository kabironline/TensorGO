//go:build cuda

package nn_test

import (
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/optim"
	"github.com/kabironline/nanograd/tensor"
	"github.com/stretchr/testify/assert"
)

func TestNewLinearUsesDefaultBackend(t *testing.T) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}

	prevBackend := backend.GetDefaultBackend()
	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)
	t.Cleanup(func() {
		backend.SetDefaultBackend(prevBackend)
	})

	layer := nn.NewLinear(4, 3)

	assert.True(t, layer.Weight.Device.IsGPU(), "linear weight should use the default GPU backend")
	assert.True(t, layer.Bias.Device.IsGPU(), "linear bias should use the default GPU backend")
}

func TestLinearForwardBackwardOnGPU(t *testing.T) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}

	prevBackend := backend.GetDefaultBackend()
	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)
	t.Cleanup(func() {
		backend.SetDefaultBackend(prevBackend)
	})

	layer := nn.NewLinear(4, 3)
	input := tensor.NewTensor([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, []int{2, 4})

	assert.True(t, input.Device.IsGPU(), "input tensor should use the default GPU backend")

	optimizer := optim.NewAdam(layer.Parameters(), 0.01)
	optimizer.ZeroGrad()

	pred := layer.Forward(input)
	loss := pred.Sum()
	loss.BackProp()
	optimizer.Step()

	assert.True(t, pred.Device.IsGPU(), "prediction tensor should remain on the GPU")
	assert.True(t, loss.Device.IsGPU(), "loss tensor should remain on the GPU")
}

func TestLinearRepeatedGPUTrainingDoesNotLeakGraph(t *testing.T) {
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}

	prevBackend := backend.GetDefaultBackend()
	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)
	t.Cleanup(func() {
		backend.SetDefaultBackend(prevBackend)
	})

	layer := nn.NewLinear(128, 64)
	input := tensor.NewTensor(make([]float32, 32*128), []int{32, 128})
	optimizer := optim.NewAdam(layer.Parameters(), 0.01)

	for step := 0; step < 64; step++ {
		optimizer.ZeroGrad()

		pred := layer.Forward(input)
		loss := pred.Sum()
		loss.BackProp()
		optimizer.Step()

		loss.ClearComputationGraph()
		pred.ClearGraph()
	}

	assert.True(t, input.Device.IsGPU(), "input tensor should remain on the GPU")
}
