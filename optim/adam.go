package optim

import (
	"math"
	"sync"

	"github.com/kabironline/nanograd/tensor"
)

type Adam struct {
	Params     []*tensor.Tensor
	LR         float64
	Beta1      float64
	Beta2      float64
	Eps        float64
	T          int         // Timestep for bias correction
	M          [][]float64 // First moment (momentum)
	V          [][]float64 // Second moment (uncentered variance)
	NumWorkers int
}

func NewAdam(params []*tensor.Tensor, lr float64) *Adam {
	m := make([][]float64, len(params))
	v := make([][]float64, len(params))
	for i, p := range params {
		m[i] = make([]float64, len(p.Data))
		v[i] = make([]float64, len(p.Data))
	}
	return &Adam{
		Params:     params,
		LR:         lr,
		Beta1:      0.9,
		Beta2:      0.999,
		Eps:        1e-8,
		T:          0,
		M:          m,
		V:          v,
		NumWorkers: 1,
	}
}

func (a *Adam) Step() {
	a.T++
	beta1Pow := math.Pow(a.Beta1, float64(a.T))
	beta2Pow := math.Pow(a.Beta2, float64(a.T))
	denom1 := 1 - beta1Pow
	denom2 := 1 - beta2Pow

	ch := make(chan int)
	var wg sync.WaitGroup

	for range a.NumWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				p := a.Params[i]
				mRow := a.M[i]
				vRow := a.V[i]
				for j := range p.Data {
					g := p.Grad[j]
					mRow[j] = a.Beta1*mRow[j] + (1-a.Beta1)*g
					vRow[j] = a.Beta2*vRow[j] + (1-a.Beta2)*g*g

					mHat := mRow[j] / denom1
					vHat := vRow[j] / denom2

					p.Data[j] -= a.LR * mHat / (math.Sqrt(vHat) + a.Eps)
				}
			}
		}()
	}

	for i := range a.Params {
		ch <- i
	}
	close(ch)
	wg.Wait()
}

func (a *Adam) ZeroGrad() {
	for _, p := range a.Params {
		for i := range p.Grad {
			p.Grad[i] = 0
		}
	}
}
