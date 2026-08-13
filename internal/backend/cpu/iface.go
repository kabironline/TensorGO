package cpu

import "github.com/kabironline/nanograd/backend"

// Compile-time proof that CPUBackend satisfies the full Backend contract.
//
// Without this, dropping or mis-signing a method still compiles inside the
// package and only fails at whatever distant site assigns it to a
// backend.Backend -- the same failure shape that let Sequential silently not
// implement nn.Module.
var _ backend.Backend = (*CPUBackend)(nil)
