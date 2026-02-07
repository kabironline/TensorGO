package mnist

import (
	"testing"

	"github.com/kabironline/nanograd/backend"
	"github.com/kabironline/nanograd/internal/backend/cuda"
	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/nn/activations"
	"github.com/kabironline/nanograd/optim"
	"github.com/kabironline/nanograd/tensor"
	"github.com/petar/GoMNIST"
	"github.com/stretchr/testify/assert"
)

func loadMNISTDataGPU(t *testing.T) (*GoMNIST.Set, *GoMNIST.Set) {
	train, test, err := GoMNIST.Load("./data")
	assert.NoError(t, err)
	assert.Equal(t, 60000, len(train.Images))
	assert.Equal(t, 10000, len(test.Images))

	// Just check that the data is loaded correctly
	assert.Equal(t, 28*28, len(train.Images[0]))
	assert.Equal(t, 28*28, len(test.Images[0]))

	return train, test
}

func prepareImageGPU(img []uint8) []float32 {
	// Normalize pixel values to [0,1] and convert to float32
	prepared := make([]float32, len(img))
	for i, v := range img {
		prepared[i] = float32(v) / 255.0
	}
	return prepared
}

func TestMNISTGPU(t *testing.T) {
	// Initialize CUDA backend
	cu, err := cuda.NewCUDABackend(0)
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}

	// Register and set CUDA as default BEFORE creating model
	backend.RegisterBackend("cuda", cu)
	backend.SetDefaultBackend(cu)

	t.Logf("CUDA backend initialized - all operations will run on GPU")

	// Build model - tensors will be automatically moved to GPU
	model := nn.NewSequential(
		nn.NewLinear(28*28, 128),
		&activations.ReLU{},
		nn.NewLinear(128, 64),
		&activations.ReLU{},
		nn.NewLinear(64, 10),
		&activations.Softmax{},
	)

	trainData, testData := loadMNISTDataGPU(t)

	numTrainSamples := len(trainData.Images)
	trainInputs := make([]float32, numTrainSamples*28*28)
	trainTargets := make([]float32, numTrainSamples*10)

	for i := range numTrainSamples {
		imgRaw := trainData.Images[i]
		img := prepareImageGPU(imgRaw)
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

	// Use Adam optimizer with learning rate
	optimizer := optim.NewAdam(model.Parameters(), 0.005)

	// Training loop with CrossEntropy loss
	const numEpochs = 5
	const batchSize = 32

	for epoch := range numEpochs {
		var totalLoss float32 = 0.0
		batchCount := 0

		// Mini-batch training
		for batch := 0; batch*batchSize < numTrainSamples; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if end > numTrainSamples {
				end = numTrainSamples
			}

			currentBatchSize := end - start

			// Create batch slices
			batchInputData := trainInputs[start*28*28 : end*28*28]
			batchTargetData := trainTargets[start*10 : end*10]

			batchInput := tensor.NewTensor(batchInputData, []int{currentBatchSize, 28 * 28})
			batchTarget := tensor.NewTensor(batchTargetData, []int{currentBatchSize, 10})

			// Forward pass
			optimizer.ZeroGrad()
			pred := model.Forward(batchInput)

			// Use CrossEntropy loss
			loss := nn.CrossEntropyLoss(pred, batchTarget)

			// Backward pass
			loss.BackProp()
			optimizer.Step()

			// For GPU, copy loss to CPU to read scalar value
			var lossVal float32
			if cu.IsGPU() {
				lossData := cu.ToCPU(loss.Data)
				lossVal = lossData[0]
			} else {
				lossVal = loss.Data[0]
			}
			totalLoss += lossVal
			batchCount++
		}

		avgLoss := totalLoss / float32(batchCount)
		t.Logf("Epoch %d, Avg Loss: %f", epoch, avgLoss)

	}

	// Evaluate on full test set
	numTestSamples := len(testData.Images)
	testInputs := make([]float32, numTestSamples*28*28)
	testTargets := make([]uint8, numTestSamples)

	for i, imgRaw := range testData.Images {
		img := prepareImageGPU(imgRaw)
		for j, v := range img {
			testInputs[i*28*28+j] = float32(v)
		}
		testTargets[i] = uint8(testData.Labels[i])
	}

	// Evaluate in batches to avoid memory issues
	correctPredictions := 0
	testBatchSize := 100
	for batch := 0; batch < (numTestSamples+testBatchSize-1)/testBatchSize; batch++ {
		start := batch * testBatchSize
		end := start + testBatchSize
		if end > numTestSamples {
			end = numTestSamples
		}

		currentBatchSize := end - start
		batchTestData := testInputs[start*28*28 : end*28*28]
		batchTest := tensor.NewTensor(batchTestData, []int{currentBatchSize, 28 * 28})

		predictions := model.Forward(batchTest)

		// Copy predictions to CPU for evaluation
		predData := predictions.Data
		if cu.IsGPU() {
			predData = cu.ToCPU(predictions.Data)
		}

		// Count correct predictions
		for i := 0; i < currentBatchSize; i++ {
			maxIdx := 0
			maxVal := predData[i*10]
			for j := 1; j < 10; j++ {
				if predData[i*10+j] > maxVal {
					maxVal = predData[i*10+j]
					maxIdx = j
				}
			}
			if maxIdx == int(testTargets[start+i]) {
				correctPredictions++
			}
		}
	}

	accuracy := float32(correctPredictions) / float32(numTestSamples) * 100.0
	t.Logf("Test Accuracy: %.2f%% (%d/%d correct)", accuracy, correctPredictions, numTestSamples)

	// Assert that we achieve at least 95% accuracy
	assert.Greater(t, accuracy, float32(95.0), "Expected test accuracy > 95%, got %.2f%%", accuracy)
}
