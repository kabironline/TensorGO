package optim

import (
	"math"

	"github.com/kabironline/nanograd/tensor"
)

type Adam struct {
	Params []*tensor.Tensor
	LR     float64
	Beta1  float64
	Beta2  float64
	Eps    float64
	T      int         // Timestep for bias correction
	M      [][]float64 // First moment (momentum)
	V      [][]float64 // Second moment (uncentered variance)
}

func NewAdam(params []*tensor.Tensor, lr float64) *Adam {
 m := make([][]float64, len(params))
 v := make([][]float64, len(params))
 for i, p := range params {
  m[i] = make([]float64, len(p.Data))
  v[i] = make([]float64, len(p.Data))
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
		for j := range p.Data {
			// Update moments
			a.M[i][j] = a.Beta1*a.M[i][j] + (1-a.Beta1)*p.Grad[j]
			a.V[i][j] = a.Beta2*a.V[i][j] + (1-a.Beta2)*p.Grad[j]*p.Grad[j]

			// Bias correction
			mHat := a.M[i][j] / (1 - math.Pow(a.Beta1, float64(a.T)))
			vHat := a.V[i][j] / (1 - math.Pow(a.Beta2, float64(a.T)))

			// Update parameter
			p.Data[j] -= a.LR * mHat / (math.Sqrt(vHat) + a.Eps)
		}
	}
}
func (a *Adam) ZeroGrad() {
	for _, p := range a.Params {
		for i := range p.Grad {
			p.Grad[i] = 0
		}
	}
}

