package cpu

import "math"

// Softmax applies the softmax activation along the last dimension of the input shape.
func (b *CPUBackend) Softmax(data []float32, shape []int) []float32 {
	if len(shape) == 0 {
		return nil
	}
	batch, classDim := b.getBatchAndClassDim(shape)
	out := b.Allocate(len(data))

	for bIdx := 0; bIdx < batch; bIdx++ {
		offset := bIdx * classDim
		// Find max for numerical stability
		maxVal := data[offset]
		for j := 1; j < classDim; j++ {
			if data[offset+j] > maxVal {
				maxVal = data[offset+j]
			}
		}

		sum := float32(0.0)
		for j := 0; j < classDim; j++ {
			e := float32(math.Exp(float64(data[offset+j] - maxVal)))
			out[offset+j] = e
			sum += e
		}

		for j := 0; j < classDim; j++ {
			out[offset+j] /= sum
		}
	}
	return out
}

// SoftmaxBackward computes the gradient of softmax
func (b *CPUBackend) SoftmaxBackward(grad, output []float32, shape []int) []float32 {
	batch, classDim := b.getBatchAndClassDim(shape)
	out := b.Allocate(len(grad))

	for bIdx := 0; bIdx < batch; bIdx++ {
		offset := bIdx * classDim
		// For each sample in batch
		for i := 0; i < classDim; i++ {
			var sum float32
			for j := 0; j < classDim; j++ {
				if i == j {
					sum += grad[offset+j] * output[offset+i] * (1 - output[offset+i])
				} else {
					sum -= grad[offset+j] * output[offset+i] * output[offset+j]
				}
			}
			out[offset+i] = sum
		}
	}
	return out
}

// LogSoftmax applies log(softmax(x))
func (b *CPUBackend) LogSoftmax(data []float32, shape []int) []float32 {
	batch, classDim := b.getBatchAndClassDim(shape)
	out := b.Allocate(len(data))

	for bIdx := 0; bIdx < batch; bIdx++ {
		offset := bIdx * classDim
		maxVal := data[offset]
		for j := 1; j < classDim; j++ {
			if data[offset+j] > maxVal {
				maxVal = data[offset+j]
			}
		}

		logSumExp := float32(0.0)
		for j := 0; j < classDim; j++ {
			logSumExp += float32(math.Exp(float64(data[offset+j] - maxVal)))
		}
		logSumExp = maxVal + float32(math.Log(float64(logSumExp)))

		for j := 0; j < classDim; j++ {
			out[offset+j] = data[offset+j] - logSumExp
		}
	}
	return out
}

func (b *CPUBackend) getBatchAndClassDim(shape []int) (int, int) {
	batch := 1
	for i := 0; i < len(shape)-1; i++ {
		batch *= shape[i]
	}
	return batch, shape[len(shape)-1]
}
