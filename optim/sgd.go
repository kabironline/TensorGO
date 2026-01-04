package optim

import (
	"sync"

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
	var wg sync.WaitGroup
	for _, p := range s.Params {
		wg.Add(1)
		go func(p *tensor.Tensor) {
			defer wg.Done()
			for i := range p.Data {
				p.Data[i] -= s.LR * p.Grad[i]
			}
		}(p)
	}
	wg.Wait()
}

// ZeroGrad sets all gradients of the parameters to zero.
func (s *SGD) ZeroGrad() {
	for _, p := range s.Params {
		for i := range p.Grad {
			p.Grad[i] = 0
		}
	}
}
