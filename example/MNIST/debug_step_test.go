package mnist

import (
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/tensor"
)

func TestDebugOneStep(t *testing.T) {
	// Initialize CUDA backend
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}

	// Register and set CUDA as default BEFORE creating model
	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)

	t.Log("CUDA backend initialized")

	// Create even simpler model - just one linear layer
	linear := nn.NewLinear(2, 2)

	// Get initial weights
	w0Init := cu.ToCPU(linear.Weight.Data)
	t.Logf("Initial weight: [[%.4f, %.4f], [%.4f, %.4f]]", w0Init[0], w0Init[1], w0Init[2], w0Init[3])

	// Create simple input and target
	input := tensor.NewTensor([]float32{1, 2}, []int{1, 2})
	target := tensor.NewTensor([]float32{0, 1}, []int{1, 2})

	// Forward
	output := linear.Forward(input) // Should call MatMulAddBias

	outData := cu.ToCPU(output.Data)
	t.Logf("Output: [%.4f, %.4f]", outData[0], outData[1])

	// Apply softmax and loss manually to debug
	prob := output.Softmax()
	probData := cu.ToCPU(prob.Data)
	t.Logf("After softmax: [%.4f, %.4f]", probData[0], probData[1])

	loss := nn.CrossEntropyLoss(prob, target)
	lossData := cu.ToCPU(loss.Data)
	t.Logf("Loss: %.4f", lossData[0])

	// Debug: Check if computation graph is set up correctly
	t.Logf("loss has %d parents", len(loss.Parents))
	t.Logf("prob has %d parents", len(prob.Parents))
	t.Logf("output has %d parents", len(output.Parents))
	t.Logf("linear.Weight.RequiresGrad = %v", linear.Weight.RequiresGrad)
	t.Logf("linear.Bias.RequiresGrad = %v", linear.Bias.RequiresGrad)
	t.Logf("output.RequiresGrad = %v", output.RequiresGrad)
	t.Logf("output.Backward != nil: %v", output.Backward != nil)
	t.Logf("prob.Backward != nil: %v", prob.Backward != nil)
	t.Logf("loss.Backward != nil: %v", loss.Backward != nil)

	// Set RequiresGrad before creating the graph (this is wrong - should be before Forward)
	// prob.RequiresGrad = true
	// linear.Weight.RequiresGrad = true
	// linear.Bias.RequiresGrad = true

	// Backward
	loss.BackProp()

	// Check if loss gradient was initialized
	if loss.Grad != nil {
		lossGrad := cu.ToCPU(loss.Grad)
		t.Logf("loss.Grad: [%.6f]", lossGrad[0])
	} else {
		t.Log("loss.Grad is nil")
	}

	// Check intermediate gradients
	if prob.Grad != nil {
		probGrad := cu.ToCPU(prob.Grad)
		t.Logf("prob.Grad: [%.6f, %.6f]", probGrad[0], probGrad[1])
	} else {
		t.Log("prob.Grad is nil")
	}

	if output.Grad != nil {
		outputGrad := cu.ToCPU(output.Grad)
		t.Logf("output.Grad: [%.6f, %.6f]", outputGrad[0], outputGrad[1])
	} else {
		t.Log("output.Grad is nil")
	}

	// Check weight gradient
	if linear.Weight.Grad != nil {
		wGrad := cu.ToCPU(linear.Weight.Grad)
		t.Logf("Weight gradient: [[%.6f, %.6f], [%.6f, %.6f]]", wGrad[0], wGrad[1], wGrad[2], wGrad[3])

		allZero := true
		for _, g := range wGrad {
			if g != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Error("Weight gradients are ALL ZERO!")
		} else {
			t.Log("✓ Weight gradients are non-zero")
		}
	} else {
		t.Error("Weight.Grad is nil!")
	}

	// Check bias gradient
	if linear.Bias.Grad != nil {
		bGrad := cu.ToCPU(linear.Bias.Grad)
		t.Logf("Bias gradient: [%.6f, %.6f]", bGrad[0], bGrad[1])
	} else {
		t.Error("Bias.Grad is nil!")
	}
}
