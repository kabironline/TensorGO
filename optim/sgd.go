package optim

import "github.com/kabironline/nanograd/tensor"

type SGD struct {
	Params []*tensor.Tensor
	LR     float64
}

func NewSGD(params []*tensor.Tensor, lr float64) *SGD {
	return &SGD{Params: params, LR: lr}
}

func (s *SGD) Step() {
	for _, p := range s.Params {
		for i := range p.Data {
			p.Data[i] -= s.LR * p.Grad[i]
		}
	}
}

// ZeroGrad sets all gradients of the parameters to zero.
func (s *SGD) ZeroGrad() {
	for _, p := range s.Params {
		for i := range p.Grad {
			p.Grad[i] = 0
		}
	}
}
