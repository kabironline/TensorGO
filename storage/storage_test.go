package storage

import "testing"

func TestFromF32RoundTrip(t *testing.T) {
	s := From([]float32{1, 2, 3})
	if s.DType() != F32 {
		t.Fatalf("dtype: want F32 got %v", s.DType())
	}
	if s.Length() != 3 {
		t.Fatalf("length: want 3 got %d (header-cast bug would give 12)", s.Length())
	}
	got := s.F32()
	if len(got) != 3 {
		t.Fatalf("F32 len: want 3 got %d", len(got))
	}
	for i, v := range []float32{1, 2, 3} {
		if got[i] != v {
			t.Fatalf("F32[%d]: want %v got %v", i, v, got[i])
		}
	}
}

func TestF32SharesBacking(t *testing.T) {
	src := []float32{5, 6, 7}
	s := From(src)
	s.F32()[1] = 99 // zero-copy: writing through Storage must mutate src
	if src[1] != 99 {
		t.Fatalf("expected zero-copy aliasing, src[1]=%v", src[1])
	}
}

func TestF32PanicsOnDtypeMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic calling F32() on an F64 storage")
		}
	}()
	From([]float64{1, 2, 3}).F32()
}

func TestEmptyStorage(t *testing.T) {
	s := From([]float32{})
	if s.Length() != 0 || s.F32() != nil {
		t.Fatalf("empty storage: length=%d f32=%v", s.Length(), s.F32())
	}
}

func TestDTypeSizes(t *testing.T) {
	cases := map[DType]int{F32: 4, F64: 8, I32: 4, I8: 1, Bool: 1, I16: 2, I64: 8}
	for dt, want := range cases {
		if dt.Size() != want {
			t.Errorf("%v.Size(): want %d got %d", dt, want, dt.Size())
		}
	}
}

func TestSetDataFromBytes(t *testing.T) {
	// 2 float32 little-endian: 1.0 = 0x3F800000, 2.0 = 0x40000000
	bytes := []byte{0, 0, 0x80, 0x3F, 0, 0, 0, 0x40}
	s := &Storage{}
	s.SetData(bytes, F32)
	if s.Length() != 2 {
		t.Fatalf("length: want 2 got %d", s.Length())
	}
	got := s.F32()
	if got[0] != 1 || got[1] != 2 {
		t.Fatalf("decoded: %v", got)
	}
}
