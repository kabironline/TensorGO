package optim

import (
	"github.com/kabironline/nanograd/tensor"
)

type Adam struct {
	Params []*tensor.Tensor
	LR     float32
	Beta1  float32
	Beta2  float32
	Eps    float32
	T      int         // Timestep for bias correction
	M      [][]float32 // First moment (momentum)
	V      [][]float32 // Second moment (uncentered variance)
}

func NewAdam(params []*tensor.Tensor, lr float32) *Adam {
	m := make([][]float32, len(params))
	v := make([][]float32, len(params))
	for i, p := range params {
		m[i] = p.Device.Allocate(len(p.Data()))
		v[i] = p.Device.Allocate(len(p.Data()))
		// Initialize with zeros
		p.Device.Fill(m[i], 0, len(m[i]))
		p.Device.Fill(v[i], 0, len(v[i]))
	}
	return &Adam{
		Params: params,
		LR:     lr,
		Beta1:  0.9,
		Beta2:  0.999,
		Eps:    1e-8,
		T:      0,
		M:      m,
		V:      v,
	}
}

func (a *Adam) Step() {
	a.T++
	for i, p := range a.Params {
		if p.Grad() == nil {
			continue
		}
		p.Device.StepAdam(p.Data(), p.Grad(), a.M[i], a.V[i], a.LR, a.Beta1, a.Beta2, a.Eps, a.T)
	}
}

func (a *Adam) ZeroGrad() {
	for _, p := range a.Params {
		if p.Grad() != nil {
			p.Device.Fill(p.Grad(), 0, len(p.Grad()))
		}
	}
}
