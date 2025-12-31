package nn

// This file contains different loss functions used in neural networks.
// MSE and CrossEntropy are implemented here.

import (
	"math"

	"github.com/kabironline/nanograd/tensor"
)

// MSELoss computes the Mean Squared Error loss between predictions and targets.
func MSELoss(predictions, targets *tensor.Tensor) *tensor.Tensor {
	if len(predictions.Data) != len(targets.Data) {
		panic("Predictions and targets must have the same length")
	}

	lossData := make([]float64, len(predictions.Data))
	for i := range predictions.Data {
		diff := predictions.Data[i] - targets.Data[i]
		lossData[i] = diff * diff
	}

	meanLoss := 0.0
	for _, v := range lossData {
		meanLoss += v
	}

	meanLoss /= float64(len(lossData))

	out := tensor.NewTensor([]float64{meanLoss}, []int{1}, predictions, targets)
	out.Backward = func() {
		for i := range predictions.Data {
			grad := 2 * (predictions.Data[i] - targets.Data[i]) / float64(len(predictions.Data))
			predictions.Grad[i] += grad
			targets.Grad[i] += -grad
		}
	}
	return out
}

// CrossEntropyLoss computes the Cross Entropy loss between predictions and targets.
// Assumes predictions are probabilities (after softmax) and targets are one-hot encoded.
func CrossEntropyLoss(predictions, targets *tensor.Tensor) *tensor.Tensor {
	if len(predictions.Data) != len(targets.Data) {
		panic("Predictions and targets must have the same length")
	}

	loss := 0.0
	for i := range predictions.Data {
		if targets.Data[i] == 1 {
			loss -= math.Log(predictions.Data[i] + 1e-15) // Adding a small value to avoid log(0)
		}
	}

	batchSize := 1
	if len(predictions.Shape) > 0 {
		batchSize = predictions.Shape[0]
	}
	meanLoss := loss / float64(batchSize)

	out := tensor.NewTensor([]float64{meanLoss}, []int{1}, predictions, targets)
	out.Backward = func() {
		scale := 1.0 / float64(batchSize)
		for i := range predictions.Data {
			grad := -targets.Data[i] / (predictions.Data[i] + 1e-15)
			predictions.Grad[i] += grad * scale
			targets.Grad[i] += -grad * scale
		}
	}
	return out

}
