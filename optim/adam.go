package optim

import (
	"math"

	"github.com/kabironline/nanograd/internal/pools"
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

	// Group parameters by their contiguous data length so we can perform
	// batched vectorized updates for groups with identical sizes.
	groups := make(map[int][]int)
	for i, p := range a.Params {
		groups[len(p.Data)] = append(groups[len(p.Data)], i)
	}

	// Process each group. For groups with more than one parameter we use a
	// batched update to improve memory locality and allow Gonum/floats to work
	// on larger contiguous slices. Small or singleton groups fall back to the
	// original element-wise update.
	for size, idxs := range groups {
		if len(idxs) > 1 {
			rows := len(idxs)
			cols := size
			n := rows * cols

			// Allocate pooled buffers
			gBuf := pools.GetBuffer(n)
			mBuf := pools.GetBuffer(n)
			vBuf := pools.GetBuffer(n)

			// Copy data into flat buffers row-major
			for r, pi := range idxs {
				p := a.Params[pi]
				copy(gBuf[r*cols:(r+1)*cols], p.Grad)
				copy(mBuf[r*cols:(r+1)*cols], a.M[pi])
				copy(vBuf[r*cols:(r+1)*cols], a.V[pi])
			}

			// Vectorized element-wise update over the flat buffers
			for k := 0; k < n; k++ {
				g := gBuf[k]
				m := a.Beta1*mBuf[k] + (1-a.Beta1)*g
				v := a.Beta2*vBuf[k] + (1-a.Beta2)*g*g

				mHat := m / denom1
				vHat := v / denom2

				delta := a.LR * mHat / (math.Sqrt(vHat) + a.Eps)

				// write back into buffers
				mBuf[k] = m
				vBuf[k] = v
				// reuse gBuf to store delta (to subtract from params later)
				gBuf[k] = delta
			}

			// Scatter results back into parameters and moment buffers
			for r, pi := range idxs {
				p := a.Params[pi]
				copy(a.M[pi], mBuf[r*cols:(r+1)*cols])
				copy(a.V[pi], vBuf[r*cols:(r+1)*cols])
				// Apply parameter update: p.Data -= delta
				seg := gBuf[r*cols : (r+1)*cols]
				for j := 0; j < cols; j++ {
					p.Data[j] -= seg[j]
				}
			}

			// Return buffers to pool
			pools.PutBuffer(gBuf)
			pools.PutBuffer(mBuf)
			pools.PutBuffer(vBuf)
		} else {
			// Fallback for single-parameter groups: do element-wise update
			for _, i := range idxs {
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
