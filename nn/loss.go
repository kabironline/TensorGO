package nn

// This file contains different loss functions used in neural networks.
// MSE and CrossEntropy are implemented here.

import (
	"math"

	"github.com/kabironline/nanograd/tensor"
)

// MSELoss computes the Mean Squared Error loss between predictions and targets.
func MSELoss(predictions, targets *tensor.Tensor) *tensor.Tensor {
	// Check logical size, not physical data length
	if tensor.TotalSize(predictions.Shape) != tensor.TotalSize(targets.Shape) {
		panic("Predictions and targets must have the same number of elements")
	}

	pContig := tensor.Contiguous(predictions)
	tContig := tensor.Contiguous(targets)

	lossData := make([]float64, len(pContig.Data))
	for i := range pContig.Data {
		diff := pContig.Data[i] - tContig.Data[i]
		lossData[i] = diff * diff
	}

	meanLoss := 0.0
	for _, v := range lossData {
		meanLoss += v
	}

	meanLoss /= float64(len(lossData))

	out := tensor.NewTensor([]float64{meanLoss}, []int{1}, predictions, targets)
	out.Backward = func() {
		n := float64(len(lossData))

		// Gradients for predictions
		gradP := make([]float64, len(lossData))
		for i := range gradP {
			gradP[i] = 2 * (pContig.Data[i] - tContig.Data[i]) / n
		}
		predictions.AccumulateGrad(gradP)

		// Gradients for targets
		gradT := make([]float64, len(lossData))
		for i := range gradT {
			gradT[i] = -gradP[i]
		}
		targets.AccumulateGrad(gradT)
	}
	return out
}

// CrossEntropyLoss computes the Cross Entropy loss between predictions and targets.
// Assumes predictions are probabilities (after softmax) and targets are one-hot encoded.
func CrossEntropyLoss(predictions, targets *tensor.Tensor) *tensor.Tensor {
	if tensor.TotalSize(predictions.Shape) != tensor.TotalSize(targets.Shape) {
		panic("Predictions and targets must have the same number of elements")
	}

	pContig := tensor.Contiguous(predictions)
	tContig := tensor.Contiguous(targets)

	loss := 0.0
	for i := range pContig.Data {
		if tContig.Data[i] == 1 {
			loss -= math.Log(pContig.Data[i] + 1e-15) // Adding a small value to avoid log(0)
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

		gradP := make([]float64, len(pContig.Data))
		gradT := make([]float64, len(tContig.Data))

		for i := range pContig.Data {
			grad := -tContig.Data[i] / (pContig.Data[i] + 1e-15)
			gradP[i] = grad * scale
			gradT[i] = -grad * scale
		}

		predictions.AccumulateGrad(gradP)
		targets.AccumulateGrad(gradT)
	}
	return out

}
