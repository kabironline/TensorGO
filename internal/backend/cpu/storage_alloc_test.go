package cpu_test

import (
	"testing"

	"github.com/kabironline/nanograd/internal/backend/cpu"
	"github.com/kabironline/nanograd/storage"
)

// TestCPUStorageManager exercises the Phase-2 Storage-typed memory methods on the
// CPU backend: AllocStorage / FreeStorage / CopyStorage.
func TestCPUStorageManager(t *testing.T) {
	b := cpu.NewCPUBackend()

	s := b.AllocStorage(4, storage.F32)
	if s.DType() != storage.F32 {
		t.Fatalf("dtype: want F32 got %v", s.DType())
	}
	if s.Length() != 4 {
		t.Fatalf("length: want 4 got %d", s.Length())
	}
	if len(s.Bytes()) != 4*storage.F32.Size() {
		t.Fatalf("bytes: want %d got %d", 4*storage.F32.Size(), len(s.Bytes()))
	}

	// freshly allocated host storage is zeroed and writable through F32()
	f := s.F32()
	for _, v := range f {
		if v != 0 {
			t.Fatalf("AllocStorage not zeroed: %v", f)
		}
	}
	copy(f, []float32{1, 2, 3, 4})

	dst := b.AllocStorage(4, storage.F32)
	b.CopyStorage(dst, s)
	for i, v := range []float32{1, 2, 3, 4} {
		if dst.F32()[i] != v {
			t.Fatalf("CopyStorage[%d]: want %v got %v", i, v, dst.F32()[i])
		}
	}

	// FreeStorage is a no-op on CPU and must not panic.
	b.FreeStorage(s)
	b.FreeStorage(dst)
}
