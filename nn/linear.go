package nn

import "github.com/kabironline/nanograd/tensor"

type Linear struct {
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	w := tensor.NewTensor(make([]float64, inFeatures*outFeatures), []int{inFeatures, outFeatures})
	b := tensor.NewTensor(make([]float64, outFeatures), []int{outFeatures})

	// Initialize weights and biases
	w.RandomInit()
	b.ZeroInit()

	return &Linear{
		Weight: w,
		Bias:   b,
	}
}

func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.MatMul(l.Weight).Add(l.Bias)
}

func (l *Linear) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{l.Weight, l.Bias}
}
