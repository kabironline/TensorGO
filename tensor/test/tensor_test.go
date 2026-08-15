package tensor_test

import (
	"fmt"
	"math"
	"testing"

	_ "github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/kabironline/nanograd/tensor"
)

// --- Unit Tests ---

func checkMatrix(a, b []float32) (bool, error) {
	if len(a) != len(b) {
		return false, fmt.Errorf("Length mismatch: %d vs %d", len(a), len(b))

	}

	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-9 {
			return false, fmt.Errorf("Value mismatch at index %d: %f vs %f", i, a[i], b[i])
		}
	}
	return true, nil
}

func TestNewTensor(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tens := tensor.NewTensor(data, shape)

	if tensor.TotalSize(tens.Shape) != 6 {
		t.Errorf("Expected total size 6, got %d", tensor.TotalSize(tens.Shape))
	}

	// Default strides for [2, 3] should be [3, 1]
	expectedStrides := []int{3, 1}
	for i := range expectedStrides {
		if tens.Strides[i] != expectedStrides[i] {
			t.Errorf("Expected stride[%d] = %d, got %d", i, expectedStrides[i], tens.Strides[i])
		}
	}

	if !tens.Contiguous() {
		t.Error("New tensor should be contiguous")
	}
}

func TestNewIdentityTensor(t *testing.T) {
	size := 4
	tens := tensor.NewIdentityTensor(size)

	if len(tens.Data()) != size*size {
		t.Errorf("Expected data length %d, got %d", size*size, len(tens.Data()))
	}

	// Check identity property
	// Only diagonal elements should be 1
	// Off-diagonal elements should be 0
	for i := range size {
		for j := range size {
			val := tens.Data()[i*size+j]
			if i == j {
				if val != 1.0 {
					t.Errorf("Expected diagonal element [%d,%d] to be 1, got %f", i, j, val)
				}
			} else {
				if val != 0.0 {
					t.Errorf("Expected off-diagonal element [%d,%d] to be 0, got %f", i, j, val)
				}
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tens := tensor.NewTensor(data, shape)

	// Transpose (2, 3) -> (3, 2)
	// Original: [[1, 2, 3], [4, 5, 6]]
	// Transposed: [[1, 4], [2, 5], [3, 6]]
	transposed := tens.Transpose([]int{1, 0})

	if transposed.Shape[0] != 3 || transposed.Shape[1] != 2 {
		t.Errorf("Expected shape [3, 2], got %v", transposed.Shape)
	}

	if transposed.Contiguous() {
		t.Error("Transposed tensor should not be contiguous")
	}

	// Verify mapping: Transposed[1, 0] should be Original[0, 1] = 2
	if transposed.At(1, 0) != tens.At(0, 1) {
		t.Errorf("Expected value at transposed [1, 0] to be %f, got %f", tens.At(0, 1), transposed.At(1, 0))
	}
}

func TestBroadcastAdd(t *testing.T) {
	// Test broadcasting: (1, 3) + (3, 1) -> (3, 3)
	a := tensor.NewTensor([]float32{1, 2, 3}, []int{1, 3})
	b := tensor.NewTensor([]float32{4, 5, 6}, []int{3, 1})

	// Result should be:
	// [[1+4, 2+4, 3+4],
	//  [1+5, 2+5, 3+5],
	//  [1+6, 2+6, 3+6]]
	// => [5, 6, 7, 6, 7, 8, 7, 8, 9]
	res := a.Add(b)

	expected := []float32{5, 6, 7, 6, 7, 8, 7, 8, 9}
	if tensor.TotalSize(res.Shape) != 9 {
		t.Fatalf("Expected result size 9, got %d", tensor.TotalSize(res.Shape))
	}

	check, err := checkMatrix(res.Data(), expected)
	if !check {
		t.Errorf("Broadcast add result mismatch: %v", err)
	}

}

func TestBroadcastMul(t *testing.T) {
	// Test broadcasting: (2, 2) * (2,) -> (2, 2)
	a := tensor.NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	b := tensor.NewTensor([]float32{10, 20}, []int{2})

	// b broadcasts to [[10, 20], [10, 20]]
	// Result: [[10, 40], [30, 80]]
	res := tensor.BroadcastMulOp(a, b)

	expected := []float32{10, 40, 30, 80}
	check, err := checkMatrix(res.Data(), expected)
	if !check {
		t.Errorf("Broadcast mul result mismatch: %v", err)
	}
}

func TestAddResultStridesContiguous(t *testing.T) {
	a := tensor.NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	aT := a.Transpose([]int{1, 0})
	b := tensor.NewTensor([]float32{10, 20, 30, 40}, []int{2, 2})

	out := aT.Add(b)
	expected := tensor.ComputeStrides(out.Shape)

	for i, v := range expected {
		if out.Strides[i] != v {
			t.Errorf("Stride[%d]: expected %d, got %d", i, v, out.Strides[i])
		}
	}
}

func TestAtSetAt(t *testing.T) {
	data := make([]float32, 6)
	tens := tensor.NewTensor(data, []int{2, 3})

	// Set a value and read it back
	tens.SetAt(42.0, 1, 2)
	if tens.At(1, 2) != 42.0 {
		t.Fatalf("Expected tens.At(1,2) == 42.0, got %f", tens.At(1, 2))
	}

	// Out-of-bounds should panic
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for out-of-bounds index")
		}
	}()
	_ = tens.At(2, 0)
}

func TestMatMul(t *testing.T) {
	// (2, 3) * (3, 2) -> (2, 2)
	a := tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := tensor.NewTensor([]float32{7, 8, 9, 10, 11, 12}, []int{3, 2})

	// [1*7+2*9+3*11, 1*8+2*10+3*12]  => [58, 64]
	// [4*7+5*9+6*11, 4*8+5*10+6*12]  => [139, 154]
	res := a.MatMul(b)

	expected := []float32{58, 64, 139, 154}

	check, err := checkMatrix(res.Data(), expected)
	if !check {
		t.Errorf("MatMul result mismatch: %v", err)
	}
}

func TestContiguous(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	tens := tensor.NewTensor(data, []int{2, 3})

	transposed := tens.Transpose([]int{1, 0})
	if transposed.Contiguous() {
		t.Fatal("Transpose should not be contiguous")
	}

	contig := tensor.Contiguous(transposed)
	if !contig.Contiguous() {
		t.Fatal("Contiguous() should return a contiguous tensor")
	}

	// Check data order in contiguous copy
	expected := []float32{1, 4, 2, 5, 3, 6}
	check, err := checkMatrix(contig.Data(), expected)
	if !check {
		t.Errorf("Contiguous data mismatch: %v", err)
	}
}

// RHS transposed: (2,3) @ (2,3)^T -> (2,2)
func TestMatMulTransposedRHS(t *testing.T) {
	a := tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	// laid out so that b^T == [[7,8],[9,10],[11,12]]
	b := tensor.NewTensor([]float32{7, 9, 11, 8, 10, 12}, []int{2, 3})

	res := a.MatMul(b.Transpose([]int{1, 0}))

	expected := []float32{58, 64, 139, 154} // shape (2,2)
	if check, err := checkMatrix(res.Data(), expected); !check {
		t.Errorf("MatMul with transposed RHS mismatch: %v", err)
	}
}

// LHS transposed: (2,3)^T @ (2,3) -> (3,3)  — what you originally wrote
func TestMatMulTransposedLHS(t *testing.T) {
	a := tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := tensor.NewTensor([]float32{7, 8, 9, 10, 11, 12}, []int{2, 3})

	res := a.Transpose([]int{1, 0}).MatMul(b)

	expected := []float32{47, 52, 57, 64, 71, 78, 81, 90, 99} // shape (3,3)
	if check, err := checkMatrix(res.Data(), expected); !check {
		t.Errorf("MatMul with transposed LHS mismatch: %v", err)
	}
}

// --- Benchmarks ---

func BenchmarkBroadcastAddFastPath(b *testing.B) {
	// Large contiguous addition (Gonum path)
	size := 1000000
	dataA := make([]float32, size)
	dataB := make([]float32, size)
	tensA := tensor.NewTensor(dataA, []int{size})
	tensB := tensor.NewTensor(dataB, []int{size})

	b.ResetTimer()
	for b.Loop() {
		tensA.Add(tensB)
	}
}

func BenchmarkBroadcastAddSlowPath(b *testing.B) {
	// Broadcasting 1000x1000 + 1x1000 (Stride Iterator path)
	dataA := make([]float32, 1000*1000)
	dataB := make([]float32, 1000)
	tensA := tensor.NewTensor(dataA, []int{1000, 1000})
	tensB := tensor.NewTensor(dataB, []int{1, 1000})

	b.ResetTimer()
	for b.Loop() {
		tensA.Add(tensB)
	}
}

func BenchmarkMatMul(b *testing.B) {
	// 256x256 matrix multiplication
	size := (28 * 28)
	dataA := make([]float32, size*size)
	dataB := make([]float32, size*size)
	tensA := tensor.NewTensor(dataA, []int{size, size})
	tensB := tensor.NewTensor(dataB, []int{size, size})

	for b.Loop() {
		tensA.MatMul(tensB)
	}
}

func BenchmarkTranspose(b *testing.B) {
	// Transpose is O(1) metadata change
	tens := tensor.NewTensor(make([]float32, 1000*1000), []int{1000, 1000})
	b.ResetTimer()
	for b.Loop() {
		tens.Transpose([]int{1, 0})
	}
}
