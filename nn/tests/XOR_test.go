package nn_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/nn/activations"
	"github.com/kabironline/nanograd/optim"
	"github.com/kabironline/nanograd/tensor"
	"github.com/stretchr/testify/assert"
)

func TestXOR(t *testing.T) {
	// 1. Data Setup
	// Inputs: [0,0], [0,1], [1,0], [1,1]
	inputs := tensor.NewTensor([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	// Targets: 0, 1, 1, 0
	targets := tensor.NewTensor([]float32{0, 1, 1, 0}, []int{4, 1})

	// 2. Model Architecture
	// 2 inputs -> 4 hidden (ReLU) -> 1 output
	model := nn.NewSequential(
		nn.NewLinear(2, 100),
		&activations.ReLU{},
		nn.NewLinear(100, 100),
		&activations.ReLU{},
		nn.NewLinear(100, 1),
		&activations.Sigmoid{},
	)
	optimizer := optim.NewAdam(model.Parameters(), 0.01)

	var lastLoss float32 = 0.0

	// 3. Training Loop
	for epoch := range 100 {
		optimizer.ZeroGrad()

		// Forward
		pred := model.Forward(inputs)

		// Loss (MSE)
		loss := nn.MSELoss(pred, targets)
		lastLoss = loss.Data[0]
		// Backward
		loss.BackProp()

		// Update
		optimizer.Step()

		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %f\n", epoch, loss.Data[0])
		}
	}

	// 4. Saving the model
	err := model.Save("./xor_model.safetensors")
	assert.NoError(t, err)

	defer func() {
		// Clean up saved model file after test
		err := os.Remove("./xor_model.safetensors")
		if err != nil {
			t.Logf("Failed to remove test model file: %v", err)
		}
	}()

	// 5. Load the model and verify predictions remain consistent
	newModel, err := nn.Load("./xor_model.safetensors")
	assert.NoError(t, err)

	newPred := newModel.Forward(inputs)
	newLoss := nn.MSELoss(newPred, targets).Data[0]

	assert.InDelta(t, lastLoss, newLoss, 1e-6, "Loaded model should produce same loss")
}

// Benchmark for XOR training
func BenchmarkXORTraining(b *testing.B) {
	// Setup same as TestXOR
	inputs := tensor.NewTensor([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	targets := tensor.NewTensor([]float32{0, 1, 1, 0}, []int{4, 1})
	model := nn.NewSequential(
		nn.NewLinear(2, 8),
		&activations.ReLU{},
		nn.NewLinear(8, 8),
		&activations.ReLU{},
		nn.NewLinear(8, 1),
		&activations.Sigmoid{},
	)
	optimizer := optim.NewAdam(model.Parameters(), 0.01)
	b.ResetTimer()
	for b.Loop() {
		optimizer.ZeroGrad()
		pred := model.Forward(inputs)
		loss := nn.MSELoss(pred, targets)
		loss.BackProp()
		optimizer.Step()
	}
}
