package mnist

import (
	"testing"

	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/optim"
	"github.com/kabironline/nanograd/tensor"
	"github.com/petar/GoMNIST"
	"github.com/stretchr/testify/assert"
)

func loadMNISTData(t *testing.T) (*GoMNIST.Set, *GoMNIST.Set) {
	train, test, err := GoMNIST.Load("./data")
	assert.NoError(t, err)
	assert.Equal(t, 60000, len(train.Images))
	assert.Equal(t, 10000, len(test.Images))

	// Just check that the data is loaded correctly
	assert.Equal(t, 28*28, len(train.Images[0]))
	assert.Equal(t, 28*28, len(test.Images[0]))

	return train, test
}
func prepareImage(img []uint8) []float32 {
	// Normalize pixel values to [0,1] and convert to float32
	prepared := make([]float32, len(img))
	for i, v := range img {
		prepared[i] = float32(v) / 255.0
	}
	return prepared
}

func TestMNIST(t *testing.T) {
	trainData, testData := loadMNISTData(t)

	// Use a subset of training data (10,000 samples) for faster training while maintaining accuracy
	numTrainSamples := 50000
	if numTrainSamples > len(trainData.Images) {
		numTrainSamples = len(trainData.Images)
	}

	trainInputs := make([]float64, numTrainSamples*28*28)
	trainTargets := make([]float64, numTrainSamples*10)

	for i := 0; i < numTrainSamples; i++ {
		imgRaw := trainData.Images[i]
		img := prepareImage(imgRaw)
		for j, v := range img {
			trainInputs[i*28*28+j] = float64(v)
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

	// Build model: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
	model := nn.NewMLP(
		28*28,
		[]int{128, 64, 10},
		[]nn.Module{
			&nn.ReLU{},
			&nn.ReLU{},
			&nn.Softmax{},
		},
	)

	// Use Adam optimizer with learning rate
	optimizer := optim.NewAdam(model.Parameters(), 0.005)

	// Training loop with CrossEntropy loss
	const numEpochs = 30
	const batchSize = 32

	for epoch := range numEpochs {
		totalLoss := 0.0
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

			totalLoss += loss.Data[0]
			batchCount++
		}

		avgLoss := totalLoss / float64(batchCount)
		if epoch%5 == 0 {
			t.Logf("Epoch %d, Avg Loss: %f", epoch, avgLoss)
		}
	}

	// Evaluate on full test set
	numTestSamples := len(testData.Images)
	testInputs := make([]float64, numTestSamples*28*28)
	testTargets := make([]uint8, numTestSamples)

	for i, imgRaw := range testData.Images {
		img := prepareImage(imgRaw)
		for j, v := range img {
			testInputs[i*28*28+j] = float64(v)
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

		// Count correct predictions
		for i := 0; i < currentBatchSize; i++ {
			maxIdx := 0
			maxVal := predictions.Data[i*10]
			for j := 1; j < 10; j++ {
				if predictions.Data[i*10+j] > maxVal {
					maxVal = predictions.Data[i*10+j]
					maxIdx = j
				}
			}
			if maxIdx == int(testTargets[start+i]) {
				correctPredictions++
			}
		}
	}

	accuracy := float64(correctPredictions) / float64(numTestSamples) * 100.0
	t.Logf("Test Accuracy: %.2f%% (%d/%d correct)", accuracy, correctPredictions, numTestSamples)

	// Assert that we achieve at least 95% accuracy
	assert.Greater(t, accuracy, 95.0, "Expected test accuracy > 95%, got %.2f%%", accuracy)
}
