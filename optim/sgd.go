package optim

import (
	"github.com/kabironline/nanograd/tensor"
)

type SGD struct {
	Params []*tensor.Tensor
	LR     float64
}

func NewSGD(params []*tensor.Tensor, lr float64) *SGD {
	return &SGD{Params: params, LR: lr}
}

func (s *SGD) Step() {
	for _, p := range s.Params {
		if p.Grad == nil {
			continue
		}
		p.Device.StepSGD(p.Data, p.Grad, s.LR)
	}
}

// ZeroGrad sets all gradients of the parameters to zero.
func (s *SGD) ZeroGrad() {
	for _, p := range s.Params {
		if p.Grad != nil {
			p.Device.Fill(p.Grad, 0, len(p.Grad))
		}
	}
}
