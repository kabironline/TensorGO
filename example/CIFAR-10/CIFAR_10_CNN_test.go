package cifar10

import (
	"fmt"
	"os"
	"testing"

	"github.com/kabironline/nanograd/nn"
	"github.com/kabironline/nanograd/nn/activations"
	"github.com/kabironline/nanograd/nn/conv"
	"github.com/kabironline/nanograd/optim"
	"github.com/kabironline/nanograd/tensor"
	"github.com/stretchr/testify/assert"
)

const (
	imageSize  = 32 * 32 * 3   // 32x32 pixels, 3 channels (RGB)
	recordSize = 1 + imageSize // 1 byte for label + image data
)

type CIFAR10Record struct {
	imageData []byte
	label     byte
}

func loadCIFAR10Data(folderPath string) ([]CIFAR10Record, []CIFAR10Record, error) {
	var trainRecords []CIFAR10Record
	var testRecords []CIFAR10Record

	// Load training data from data_batch_1.bin to data_batch_5.bin
	for i := 1; i <= 5; i++ {
		filePath := fmt.Sprintf("%s/data_batch_%d.bin", folderPath, i)
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, nil, fmt.Errorf("error reading file %s: %v", filePath, err)
		}

		if len(data)%recordSize != 0 {
			return nil, nil, fmt.Errorf("file size of %s is not a multiple of the record size", filePath)
		}

		numImages := len(data) / recordSize
		for j := 0; j < numImages; j++ {
			start := j * recordSize
			end := start + recordSize
			record := data[start:end]

			label := record[0]
			imageData := make([]byte, len(record[1:]))
			copy(imageData, record[1:])

			trainRecords = append(trainRecords, CIFAR10Record{
				imageData: imageData,
				label:     label,
			})
		}
	}

	// Load test data from test_batch.bin
	testFilePath := fmt.Sprintf("%s/test_batch.bin", folderPath)
	testData, err := os.ReadFile(testFilePath)
	if err != nil {
		return nil, nil, fmt.Errorf("error reading file %s: %v", testFilePath, err)
	}

	if len(testData)%recordSize != 0 {
		return nil, nil, fmt.Errorf("file size of %s is not a multiple of the record size", testFilePath)
	}

	numTestImages := len(testData) / recordSize
	for j := 0; j < numTestImages; j++ {
		start := j * recordSize
		end := start + recordSize
		record := testData[start:end]

		label := record[0]
		imageData := make([]byte, len(record[1:]))
		copy(imageData, record[1:])

		testRecords = append(testRecords, CIFAR10Record{
			imageData: imageData,
			label:     label,
		})
	}

	return trainRecords, testRecords, nil
}

func TestCIFAR10_CNN(t *testing.T) {
	trainData, testData, err := loadCIFAR10Data("./data")
	assert.NoError(t, err)

	assert.Equal(t, 50000, len(trainData))
	assert.Equal(t, 10000, len(testData))
	t.Logf("Loaded %d training samples and %d test samples", len(trainData), len(testData))
	// Prepare training data
	numTrainSamples := 50000
	if numTrainSamples > len(trainData) {
		numTrainSamples = len(trainData)
	}

	trainInputs := make([]float64, numTrainSamples*3*32*32)
	trainTargets := make([]float64, numTrainSamples*10)

	for i := 0; i < numTrainSamples; i++ {
		imgData := trainData[i].imageData
		for j, v := range imgData {
			trainInputs[i*3*32*32+j] = float64(v) / 255.0
		}
		label := trainData[i].label
		for k := 0; k < 10; k++ {
			if int(label) == k {
				trainTargets[i*10+k] = 1.0
			} else {
				trainTargets[i*10+k] = 0.0
			}
		}
	}

	// Build CNN model:
	// [N, 3, 32, 32] -> Conv2d(3, 32, 3, 3) -> ReLU -> MaxPool2d(2, 2, 0) -> [N, 32, 16, 16]
	// [N, 32, 16, 16] -> Conv2d(32, 64, 3, 3) -> ReLU -> MaxPool2d(2, 2, 0) -> [N, 64, 8, 8]
	// [N, 64, 8, 8] -> Flatten -> [N, 4096]
	// [N, 4096] -> Linear(4096, 256) -> ReLU -> Linear(256, 10) -> Softmax
	model := nn.NewSequential(
		conv.NewConv2D(3, 32, 3, 3),
		&activations.ReLU{},
		conv.NewMaxPool2D(2, 2, 0),
		conv.NewConv2D(32, 64, 3, 3),
		&activations.ReLU{},
		conv.NewMaxPool2D(2, 2, 0),
		conv.NewFlatten(),
		nn.NewLinear(4096, 256),
		&activations.ReLU{},
		nn.NewLinear(256, 10),
		&activations.Softmax{},
	)

	// Use Adam optimizer
	optimizer := optim.NewAdam(model.Parameters(), 0.001)

	// Training loop
	const numEpochs = 3
	const batchSize = 64

	fmt.Printf("Starting CNN training on %d samples...\n", numTrainSamples)

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		batchCount := 0

		for batch := 0; batch*batchSize < numTrainSamples; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if end > numTrainSamples {
				end = numTrainSamples
			}

			currentBatchSize := end - start
			batchInputData := trainInputs[start*3*32*32 : end*3*32*32]
			batchTargetData := trainTargets[start*10 : end*10]

			// Input shape: [Batch, Channels, Height, Width]
			batchInput := tensor.NewTensor(batchInputData, []int{currentBatchSize, 3, 32, 32})
			batchTarget := tensor.NewTensor(batchTargetData, []int{currentBatchSize, 10})

			// Forward pass
			output := model.Forward(batchInput)

			// Loss calculation (CrossEntropy)
			loss := nn.CrossEntropyLoss(output, batchTarget)
			totalLoss += loss.Data[0]

			// Backward pass
			optimizer.ZeroGrad()
			loss.BackProp()

			// Update weights
			optimizer.Step()

			batchCount++
			if batchCount%20 == 0 {
				fmt.Printf("Epoch %d, Batch %d, Loss: %.4f\n", epoch+1, batchCount, loss.Data[0])
			}
		}
		if batchCount > 0 {
			fmt.Printf("Epoch %d completed. Average Loss: %.4f\n", epoch+1, totalLoss/float64(batchCount))
		} else {
			fmt.Printf("Epoch %d completed. No batches processed.\n", epoch+1)
		}
	}

	// Evaluation
	fmt.Println("Evaluating on test data...")
	numTestSamples := 1000
	if numTestSamples > len(testData) {
		numTestSamples = len(testData)
	}
	correct := 0

	for i := 0; i < numTestSamples; i++ {
		imgData := testData[i].imageData
		imgDataFloat := make([]float64, 3*32*32)
		for j, v := range imgData {
			imgDataFloat[j] = float64(v) / 255.0
		}

		input := tensor.NewTensor(imgDataFloat, []int{1, 3, 32, 32})
		output := model.Forward(input)

		// Find predicted class
		pred := 0
		maxVal := output.Data[0]
		for k := 1; k < 10; k++ {
			if output.Data[k] > maxVal {
				maxVal = output.Data[k]
				pred = k
			}
		}

		if int(testData[i].label) == pred {
			correct++
		}
	}

	accuracy := float64(correct) / float64(numTestSamples) * 100
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy)
	assert.Greater(t, accuracy, 20.0, "Accuracy should be at least 20%")
}
