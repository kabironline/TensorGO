package nn

// This file contains different loss functions used in neural networks.
// MSE and CrossEntropy are implemented here.

import (
	"github.com/kabironline/nanograd/tensor"
)

// MSELoss computes the Mean Squared Error loss between predictions and targets.
func MSELoss(predictions, targets *tensor.Tensor) *tensor.Tensor {
	targets = targets.To(predictions.Device)
	diff := predictions.Sub(targets)
	sq := diff.Square()
	return sq.Mean()
}

// CrossEntropyLoss computes the Cross Entropy loss between predictions and targets.
// Assumes predictions are probabilities (after softmax) and targets are one-hot encoded.
func CrossEntropyLoss(predictions, targets *tensor.Tensor) *tensor.Tensor {
	// loss = -(targets * log(predictions + eps)).Sum() / batchSize
	targets = targets.To(predictions.Device)

	// predictions + eps
	var eps float32 = 1e-15
	pPlusEps := predictions.AddScalar(eps)
	logP := pPlusEps.Log()

	mul := targets.Mul(logP)
	sum := mul.Sum()

	// negative sum
	negSum := sum.MulScalar(-1.0)

	var batchSize float32 = 1.0
	if len(predictions.Shape) > 0 {
		batchSize = float32(predictions.Shape[0])
	}

	return negSum.DivScalar(batchSize)
}
