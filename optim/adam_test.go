package optim

import (
	"math"
	"testing"

	_ "github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/kabironline/nanograd/tensor"
)

// TestAdamBatchEquivalence verifies that the batched Adam update produces the
// same parameter updates as the element-wise (reference) implementation.
func TestAdamBatchEquivalence(t *testing.T) {
	// Create two parameters with same length to force batching
	p1 := tensor.NewTensor(make([]float64, 8), []int{8})
	p2 := tensor.NewTensor(make([]float64, 8), []int{8})

	// Allocate gradient buffers and fill grads with deterministic values
	p1.AllocGrad()
	p2.AllocGrad()
	for i := range p1.Data {
		p1.Grad[i] = float64(i + 1)
		p2.Grad[i] = float64(i + 2)
		p1.Data[i] = float64(i)
		p2.Data[i] = float64(i * 2)
	}

	// Create optimizer and clone state for reference
	params := []*tensor.Tensor{p1, p2}
	opt := NewAdam(params, 0.01)

	// Keep copies for reference computation
	refData1 := append([]float64(nil), p1.Data...)
	refData2 := append([]float64(nil), p2.Data...)
	refM1 := append([]float64(nil), opt.M[0]...)
	refM2 := append([]float64(nil), opt.M[1]...)
	refV1 := append([]float64(nil), opt.V[0]...)
	refV2 := append([]float64(nil), opt.V[1]...)

	// Run Step()
	opt.Step()

	// Compute expected updates using reference element-wise formula
	tStep := opt.T
	beta1Pow := math.Pow(opt.Beta1, float64(tStep))
	beta2Pow := math.Pow(opt.Beta2, float64(tStep))
	denom1 := 1 - beta1Pow
	denom2 := 1 - beta2Pow

	for j := range refData1 {
		g1 := p1.Grad[j]
		refM1[j] = opt.Beta1*refM1[j] + (1-opt.Beta1)*g1
		refV1[j] = opt.Beta2*refV1[j] + (1-opt.Beta2)*g1*g1
		mHat := refM1[j] / denom1
		vHat := refV1[j] / denom2
		refData1[j] -= opt.LR * mHat / (math.Sqrt(vHat) + opt.Eps)

		g2 := p2.Grad[j]
		refM2[j] = opt.Beta1*refM2[j] + (1-opt.Beta1)*g2
		refV2[j] = opt.Beta2*refV2[j] + (1-opt.Beta2)*g2*g2
		mHat2 := refM2[j] / denom1
		vHat2 := refV2[j] / denom2
		refData2[j] -= opt.LR * mHat2 / (math.Sqrt(vHat2) + opt.Eps)
	}

	// Compare
	for j := range p1.Data {
		if math.Abs(p1.Data[j]-refData1[j]) > 1e-12 {
			t.Fatalf("p1 mismatch at %d: got %v expected %v", j, p1.Data[j], refData1[j])
		}
		if math.Abs(p2.Data[j]-refData2[j]) > 1e-12 {
			t.Fatalf("p2 mismatch at %d: got %v expected %v", j, p2.Data[j], refData2[j])
		}
	}
}

func BenchmarkAdamStep_Batched(b *testing.B) {
	// Create many params with identical sizes to trigger batching
	numParams := 128
	size := 256
	params := make([]*tensor.Tensor, numParams)
	for i := 0; i < numParams; i++ {
		t := tensor.NewTensor(make([]float64, size), []int{size})
		t.AllocGrad()
		for j := 0; j < size; j++ {
			t.Grad[j] = float64(j%10 + 1)
			t.Data[j] = float64(j)
		}
		params[i] = t
	}

	opt := NewAdam(params, 0.001)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		opt.Step()
	}
}
