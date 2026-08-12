package tensor

import "github.com/kabironline/nanograd/storage"

// Storage moved to its own package so `backend` can allocate it without an import
// cycle (backend <- tensor). These transitional re-exports let existing
// tensor-package code keep referring to Storage / StorageFrom / dtype constants
// unqualified. New code should prefer the `storage` package directly.
type (
	Storage = storage.Storage
	DType   = storage.DType
)

const (
	F64  = storage.F64
	F32  = storage.F32
	I64  = storage.I64
	I32  = storage.I32
	I16  = storage.I16
	I8   = storage.I8
	Bool = storage.Bool
)

// StorageFrom forwards to storage.From, preserving the existing call sites.
func StorageFrom[T storage.Numeric](data []T) *Storage { return storage.From(data) }
