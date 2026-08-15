//go:build examples

package mnist

import (
	"fmt"
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/nn/activations"
	"github.com/kabironline/nanograd/nn/conv"
	"github.com/kabironline/nanograd/optim"
	"github.com/kabironline/nanograd/tensor"
	"github.com/petar/GoMNIST"
	"github.com/stretchr/testify/assert"
)

func prepareImageForModel(img []uint8) []float32 {
	// Normalize pixel values to [0,1] and convert to float32
	prepared := make([]float32, len(img))
	for i, v := range img {
		prepared[i] = float32(v) / 255.0
	}
	return prepared
}

func TestMNISTCNN(t *testing.T) {

	// intialize backend (CPU)
	backend.SetDefaultBackend(cpu.NewCPUBackend())

	trainData, testData, err := GoMNIST.Load("./data")
	if err != nil {
		t.Skipf("dataset not available in %s (%v); "+
			"this canary needs the dataset downloaded locally", "./data", err)
	}
	// Use a subset of training data for faster training
	numTrainSamples := 50000
	if numTrainSamples > len(trainData.Images) {
		numTrainSamples = len(trainData.Images)
	}

	trainInputs := make([]float32, numTrainSamples*1*28*28)
	trainTargets := make([]float32, numTrainSamples*10)

	for i := 0; i < numTrainSamples; i++ {
		imgRaw := trainData.Images[i]
		img := prepareImageForModel(imgRaw)
		for j, v := range img {
			trainInputs[i*28*28+j] = float32(v)
		}
		label := trainData.Labels[i]
		for k := range 10 {
			if int(label) == k {
				trainTargets[i*10+k] = 1.0
			} else {
				trainTargets[i*10+k] = 0.0
			}
		}
	}

	// Build CNN model:
	// [N, 1, 28, 28] -> Conv2d(1, 16, 3, 3) -> ReLU -> MaxPool2d(2, 2, 0) -> [N, 16, 14, 14]
	// [N, 16, 14, 14] -> Conv2d(16, 32, 3, 3) -> ReLU -> MaxPool2d(2, 2, 0) -> [N, 32, 7, 7]
	// [N, 32, 7, 7] -> Flatten -> [N, 1568]
	// [N, 1568] -> Linear(1568, 128) -> ReLU -> Linear(128, 10) -> Softmax
	model := nn.NewSequential(
		conv.NewConv2D(1, 16, 3, 3),
		&activations.ReLU{},
		conv.NewMaxPool2D(2, 2, 0),
		conv.NewConv2D(16, 32, 3, 3),
		&activations.ReLU{},
		conv.NewMaxPool2D(2, 2, 0),
		conv.NewFlatten(),
		nn.NewLinear(1568, 128),
		&activations.ReLU{},
		nn.NewLinear(128, 10),
		&activations.Softmax{},
	)

	// Use Adam optimizer
	optimizer := optim.NewAdam(model.Parameters(), 0.001)

	// Training loop
	const numEpochs = 3
	const batchSize = 64

	fmt.Printf("Starting CNN training on %d samples...\n", numTrainSamples)

	for epoch := range numEpochs {
		var totalLoss float32 = 0.0
		batchCount := 0

		for batch := 0; batch*batchSize < numTrainSamples; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if end > numTrainSamples {
				end = numTrainSamples
			}

			currentBatchSize := end - start
			batchInputData := trainInputs[start*28*28 : end*28*28]
			batchTargetData := trainTargets[start*10 : end*10]

			// Input shape: [Batch, Channels, Height, Width]
			batchInput := tensor.NewTensor(batchInputData, []int{currentBatchSize, 1, 28, 28})
			batchTarget := tensor.NewTensor(batchTargetData, []int{currentBatchSize, 10})

			// Forward pass
			output := model.Forward(batchInput)

			// Loss calculation (CrossEntropy)
			loss := nn.CrossEntropyLoss(output, batchTarget)
			totalLoss += loss.Data()[0]

			// Backward pass
			optimizer.ZeroGrad()
			loss.BackProp()

			// Update weights
			optimizer.Step()

			batchCount++
			if batchCount%20 == 0 {
				fmt.Printf("Epoch %d, Batch %d, Loss: %.4f\n", epoch+1, batchCount, loss.Data()[0])
			}
		}
		fmt.Printf("Epoch %d completed. Average Loss: %.4f\n", epoch+1, totalLoss/float32(batchCount))
	}

	// Evaluation
	fmt.Println("Evaluating on test data...")
	numTestSamples := 1000
	correct := 0

	for i := 0; i < numTestSamples; i++ {
		imgRaw := testData.Images[i]
		img := prepareImageForModel(imgRaw)
		imgData := make([]float32, 28*28)
		for j, v := range img {
			imgData[j] = float32(v)
		}

		input := tensor.NewTensor(imgData, []int{1, 1, 28, 28})
		output := model.Forward(input)

		// Find predicted class
		pred := 0
		maxVal := output.Data()[0]
		for k := 1; k < 10; k++ {
			if output.Data()[k] > maxVal {
				maxVal = output.Data()[k]
				pred = k
			}
		}

		if int(testData.Labels[i]) == pred {
			correct++
		}
	}

	accuracy := float32(correct) / float32(numTestSamples) * 100
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy)
	assert.Greater(t, accuracy, float32(80.0), "Accuracy should be at least 80%")
}
