package tensor_test

import (
	"testing"

	_ "github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/kabironline/nanograd/tensor"
)

func TestAddScalar(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	tens := tensor.NewTensor(data, shape)

	res := tens.AddScalar(10.5)

	expected := []float64{11.5, 12.5, 13.5, 14.5}
	if len(res.Data) != len(expected) {
		t.Fatalf("Expected data length %d, got %d", len(expected), len(res.Data))
	}

	for i, v := range expected {
		if res.Data[i] != v {
			t.Errorf("At index %d: expected %f, got %f", i, v, res.Data[i])
		}
	}

	// Ensure original tensor is not modified
	if tens.Data[0] != 1 {
		t.Error("Original tensor was modified by AddScalar")
	}
}

func TestSubScalar(t *testing.T) {
	data := []float64{10, 20, 30, 40}
	shape := []int{2, 2}
	tens := tensor.NewTensor(data, shape)

	res := tens.SubScalar(5)

	expected := []float64{5, 15, 25, 35}
	for i, v := range expected {
		if res.Data[i] != v {
			t.Errorf("At index %d: expected %f, got %f", i, v, res.Data[i])
		}
	}
}

func TestMulScalar(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	shape := []int{2, 2}
	tens := tensor.NewTensor(data, shape)

	res := tens.MulScalar(3)

	expected := []float64{3, 6, 9, 12}
	for i, v := range expected {
		if res.Data[i] != v {
			t.Errorf("At index %d: expected %f, got %f", i, v, res.Data[i])
		}
	}
}

func TestDivScalar(t *testing.T) {
	data := []float64{10, 20, 30, 40}
	shape := []int{2, 2}
	tens := tensor.NewTensor(data, shape)

	res := tens.DivScalar(10)

	expected := []float64{1, 2, 3, 4}
	for i, v := range expected {
		if res.Data[i] != v {
			t.Errorf("At index %d: expected %f, got %f", i, v, res.Data[i])
		}
	}
}

func TestScalarOpsPreserveShape(t *testing.T) {
	shape := []int{2, 3, 4}
	data := make([]float64, 2*3*4)
	tens := tensor.NewTensor(data, shape)

	ops := []func(float64) *tensor.Tensor{
		tens.AddScalar,
		tens.SubScalar,
		tens.MulScalar,
		tens.DivScalar,
	}

	for _, op := range ops {
		res := op(1.0)
		if len(res.Shape) != len(shape) {
			t.Errorf("Expected rank %d, got %d", len(shape), len(res.Shape))
		}
		for i := range shape {
			if res.Shape[i] != shape[i] {
				t.Errorf("Expected dim %d to be %d, got %d", i, shape[i], res.Shape[i])
			}
		}
	}
}

func BenchmarkAddScalar(b *testing.B) {
	// Large contiguous add scalar
	size := 1000000
	data := make([]float64, size)
	tens := tensor.NewTensor(data, []int{size})

	b.ResetTimer()
	for b.Loop() {
		tens.AddScalar(1.2345)
	}
}

func BenchmarkSubScalar(b *testing.B) {
	// Large contiguous sub scalar
	size := 1000000
	data := make([]float64, size)
	tens := tensor.NewTensor(data, []int{size})

	b.ResetTimer()
	for b.Loop() {
		tens.SubScalar(2.3456)
	}
}

func BenchmarkMulScalar(b *testing.B) {
	// Large contiguous mul scalar
	size := 1000000
	data := make([]float64, size)
	tens := tensor.NewTensor(data, []int{size})

	b.ResetTimer()
	for b.Loop() {
		tens.MulScalar(3.4567)
	}
}

func BenchmarkDivScalar(b *testing.B) {
	// Large contiguous div scalar
	size := 1000000
	data := make([]float64, size)
	tens := tensor.NewTensor(data, []int{size})

	b.ResetTimer()
	for b.Loop() {
		tens.DivScalar(4.5678)
	}
}
