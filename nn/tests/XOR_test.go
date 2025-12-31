package nn_test

import (
	"fmt"
	"testing"

	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/optim"
	"github.com/kabironline/nanograd/tensor"
)

func TestXOR(t *testing.T) {
	// 1. Data Setup
	// Inputs: [0,0], [0,1], [1,0], [1,1]
	inputs := tensor.NewTensor([]float64{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	// Targets: 0, 1, 1, 0
	targets := tensor.NewTensor([]float64{0, 1, 1, 0}, []int{4, 1})

	// 2. Model Architecture
	// 2 inputs -> 4 hidden (ReLU) -> 1 output
	model := nn.NewMLP(
		2,
		[]int{100, 100, 1},
		[]nn.Module{
			&nn.ReLU{},
			&nn.ReLU{},
			&nn.Sigmoid{},
		},
	)
	optimizer := optim.NewAdam(model.Parameters(), 0.01)

	// 3. Training Loop
	for epoch := range 100 {
		optimizer.ZeroGrad()

		// Forward
		pred := model.Forward(inputs)

		// Loss (MSE)
		loss := nn.MSELoss(pred, targets)

		// Backward
		loss.BackProp()

		// Update
		optimizer.Step()

		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %f\n", epoch, loss.Data[0])
		}
	}
}

// Benchmark for XOR training
func BenchmarkXORTraining(b *testing.B) {
	// Setup same as TestXOR
	inputs := tensor.NewTensor([]float64{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	targets := tensor.NewTensor([]float64{0, 1, 1, 0}, []int{4, 1})
	model := nn.NewMLP(
		2,
		[]int{8, 8, 1},
		[]nn.Module{
			&nn.ReLU{},
			&nn.ReLU{},
			&nn.Sigmoid{},
		},
	)
	optimizer := optim.NewSGD(model.Parameters(), 0.01)
	b.ResetTimer()
	for b.Loop() {
		optimizer.ZeroGrad()
		pred := model.Forward(inputs)
		loss := nn.MSELoss(pred, targets)
		loss.BackProp()
		optimizer.Step()
	}
}
