// Package storage defines the typed, device-agnostic buffer that underlies a
// Tensor. It depends on neither `tensor` nor `backend`, so both can import it
// without an import cycle — which is what lets a backend allocate a *Storage.
package storage

import "unsafe"

// DType represents the element type of the data held by a Storage.
type DType int

const (
	F64 DType = iota
	F32
	I64
	I32
	I16
	I8
	Bool
)

// Numeric is every element type Storage can hold. Add a dtype here once, ever.
type Numeric interface {
	~float32 | ~float64 | ~int32 | ~int8
}

// dtypeOf maps a Go type to its DType tag. The ONE place the mapping lives.
func dtypeOf[T Numeric]() DType {
	var z T
	switch any(z).(type) {
	case float32:
		return F32
	case float64:
		return F64
	case int32:
		return I32
	case int8:
		return I8
	default:
		panic("storage: unsupported element type")
	}
}

func (d DType) String() string {
	switch d {
	case F64:
		return "F64"
	case F32:
		return "F32"
	case I64:
		return "I64"
	case I32:
		return "I32"
	case I16:
		return "I16"
	case I8:
		return "I8"
	case Bool:
		return "Bool"
	default:
		return "Invalid"
	}
}

func (d DType) Size() int {
	switch d {
	case F64:
		return 8
	case F32:
		return 4
	case I64:
		return 8
	case I32:
		return 4
	case I16:
		return 2
	case I8:
		return 1
	case Bool:
		return 1
	default:
		panic("storage: unknown DType")
	}
}

// Buffer is a block of memory that may live in host RAM or on a device. It is the
// one field that knows *where* a Storage's bytes physically are; everything else
// in Storage (dtype, length) is just a label.
type Buffer interface {
	Bytes() []byte // host: the backing slice. device: PANICS (copy to host first).
	Len() int      // size in bytes
	Free()         // host: no-op (GC reclaims it). device: return memory to the pool.
}

// hostBuffer is a GC-owned CPU byte slice. Free is a no-op — the GC handles it.
type hostBuffer struct{ b []byte }

func (h hostBuffer) Bytes() []byte { return h.b }
func (h hostBuffer) Len() int      { return len(h.b) }
func (h hostBuffer) Free()         {}

// NewHost wraps a host byte slice as a Buffer.
func NewHost(b []byte) Buffer { return hostBuffer{b: b} }

// Storage is a flat, typed buffer: a Buffer (the bytes, wherever they live) plus a
// label (dtype + element count) describing how to read them.
type Storage struct {
	buf    Buffer
	dtype  DType
	length int
}

// New builds a Storage around an existing Buffer. Backends use this to wrap their
// own (host or device) buffer implementations.
func New(buf Buffer, dt DType, numel int) *Storage {
	return &Storage{buf: buf, dtype: dt, length: numel}
}

func (s *Storage) Buffer() Buffer { return s.buf }
func (s *Storage) DType() DType   { return s.dtype }
func (s *Storage) Length() int    { return s.length }

// Bytes returns the raw bytes. Valid only for host-resident storage; a device
// Buffer panics (its bytes are not addressable from Go).
func (s *Storage) Bytes() []byte { return s.buf.Bytes() }

// Free releases the underlying buffer (no-op for host, returns to pool for device).
func (s *Storage) Free() {
	if s.buf != nil {
		s.buf.Free()
	}
}

func (s *Storage) assertDType(expected DType) {
	if s.dtype != expected {
		panic("storage: DType mismatch expecting " + expected.String() + " but got " + s.dtype.String())
	}
}

// From builds a host Storage from ANY numeric slice — one function, all dtypes.
// The resulting Storage shares the input's backing array (zero copy).
func From[T Numeric](data []T) *Storage {
	dt := dtypeOf[T]()
	if len(data) == 0 {
		return &Storage{buf: NewHost(nil), dtype: dt, length: 0}
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*dt.Size())
	return &Storage{buf: NewHost(bytes), dtype: dt, length: len(data)}
}

// SetData replaces the buffer contents with host bytes (used e.g. when loading).
func (s *Storage) SetData(data []byte, dtype DType) {
	s.buf = NewHost(data)
	s.dtype = dtype
	s.length = len(data) / dtype.Size()
}

// F32 reinterprets the host byte buffer as []float32 — the ONE place the concrete
// type is reintroduced. Panics on a dtype mismatch or device-resident storage.
func (s *Storage) F32() []float32 {
	s.assertDType(F32)
	if s.length == 0 {
		return nil
	}
	b := s.buf.Bytes()
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), s.length)
}
