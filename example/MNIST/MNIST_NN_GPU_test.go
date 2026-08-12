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
	prevBackend := backend.GetDefaultBackend()
	backend.SetDefaultBackend(cu)
	defer backend.SetDefaultBackend(prevBackend)

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

	// GPU-optimized batch size: Larger batches amortize kernel launch overhead
	// and maximize GPU parallelism. CPU uses batch size 32.
	batchSize := 128

	// Use Adam optimizer with learning rate
	optimizer := optim.NewAdam(model.Parameters(), 0.005)

	// PyTorch/TensorFlow Pattern: Pre-allocate persistent GPU buffers for batch data
	// These buffers are reused every iteration, avoiding cudaMalloc() stalls
	var batchInputBuffer, batchTargetBuffer *tensor.Tensor
	if cu.IsGPU() {
		// Allocate max batch size buffers on GPU once
		batchInputBuffer = tensor.NewEmptyTensor([]int{batchSize, 28 * 28}, cu)
		batchTargetBuffer = tensor.NewEmptyTensor([]int{batchSize, 10}, cu)
	}

	// Training loop with CrossEntropy loss
	const numEpochs = 15
	const samplingInterval = 10 // Sample loss every N batches (PyTorch-style)

	for epoch := range numEpochs {
		var sampledLossSum float32 = 0.0
		sampledCount := 0

		// Mini-batch training - async style
		for batch := 0; batch*batchSize < numTrainSamples; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if end > numTrainSamples {
				end = numTrainSamples
			}

			currentBatchSize := end - start

			// Get host batch slices
			batchInputData := trainInputs[start*28*28 : end*28*28]
			batchTargetData := trainTargets[start*10 : end*10]

			var batchInput, batchTarget *tensor.Tensor

			if cu.IsGPU() && currentBatchSize == batchSize {
				// TRUE PyTorch pattern: Reuse pre-allocated buffer tensors directly
				// Just copy new data into them - NO new tensor allocation!
				cu.WriteToDevice(batchInputBuffer.Data(), batchInputData)
				cu.WriteToDevice(batchTargetBuffer.Data(), batchTargetData)

				// Use the buffer tensors AS-IS (they were properly created with NewEmptyTensor)
				batchInput = batchInputBuffer
				batchTarget = batchTargetBuffer
			} else {
				// Fallback for last batch or CPU mode
				batchInput = tensor.NewTensor(batchInputData, []int{currentBatchSize, 28 * 28})
				batchTarget = tensor.NewTensor(batchTargetData, []int{currentBatchSize, 10})
			}

			// Forward pass
			optimizer.ZeroGrad()
			pred := model.Forward(batchInput)

			// Use CrossEntropy loss
			loss := nn.CrossEntropyLoss(pred, batchTarget)

			// Backward pass
			loss.BackProp()
			optimizer.Step()

			// Sample loss reading (PyTorch-style) - don't sync on every batch
			// Only read loss values occasionally to avoid GPU stalls
			if batch%samplingInterval == 0 {
				var lossVal float32
				if cu.IsGPU() {
					lossData := cu.ToCPU(loss.Data())
					lossVal = lossData[0]
				} else {
					lossVal = loss.Data()[0]
				}
				sampledLossSum += lossVal
				sampledCount++
			}

			// Clean up intermediate computations (keep model parameters)
			loss.ClearComputationGraph()
			pred.ClearGraph()

			// Only clean batch tensors if they were newly allocated (fallback path)
			// Don't clean the persistent buffer tensors - we're reusing them!
			if currentBatchSize != batchSize {
				batchInput.ClearGraph()
				batchTarget.ClearGraph()
			}
		}

		// Calculate approximate average loss from sampled batches
		avgLoss := float32(0)
		if sampledCount > 0 {
			avgLoss = sampledLossSum / float32(sampledCount)
		}
		t.Logf("Epoch %d, Avg Loss: %f", epoch, avgLoss)

		// Sync GPU at end of epoch to ensure all work is done
		cu.Sync()
	}

	// Free persistent batch buffers
	if cu.IsGPU() {
		batchInputBuffer.ClearGraph()
		batchTargetBuffer.ClearGraph()
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

	// Preload all test data to GPU (PyTorch-style batching)
	var deviceTestInputs []float32
	if cu.IsGPU() {
		deviceTestInputs = cu.ToDevice(testInputs)
	} else {
		deviceTestInputs = testInputs
	}

	// Process predictions in batches and collect results on GPU
	correctPredictions := 0
	testBatchSize := 100
	for batch := 0; batch < (numTestSamples+testBatchSize-1)/testBatchSize; batch++ {
		start := batch * testBatchSize
		end := start + testBatchSize
		if end > numTestSamples {
			end = numTestSamples
		}

		currentBatchSize := end - start

		// Create tensors from preloaded device data (no additional copy)
		inSlice := deviceTestInputs[start*28*28 : end*28*28]
		batchTest := tensor.FromData(
			inSlice,
			[]int{currentBatchSize, 28 * 28},
			cu,
			false, // requiresGrad
		)

		predictions := model.Forward(batchTest)

		// Copy predictions to CPU for evaluation
		predData := predictions.Data()
		if cu.IsGPU() {
			predData = cu.ToCPU(predictions.Data())
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

		// Cleanup prediction tensors (not input, since it's a view of preloaded buffer)
		predictions.ClearGraph()
	}

	// Cleanup preloaded test data
	if cu.IsGPU() {
		cu.Free(deviceTestInputs)
	}

	accuracy := float32(correctPredictions) / float32(numTestSamples) * 100.0
	t.Logf("Test Accuracy: %.2f%% (%d/%d correct)", accuracy, correctPredictions, numTestSamples)

	// Assert that we achieve at least 95% accuracy
	assert.Greater(t, accuracy, float32(95.0), "Expected test accuracy > 95%, got %.2f%%", accuracy)
}
