package cpu

import (
	"math/rand"
	"sync"
)

// The CPU backend owns its random source rather than using the global one in
// math/rand. Go seeds that global randomly per process, so weight initialisation
// differed on every run and no accuracy result was reproducible — which also
// made the MNIST/CIFAR gates inherently flaky and their failures impossible to
// reproduce.
//
// Guarded by a mutex because rand.Rand is not safe for concurrent use and the
// backend runs ops across a worker pool.
var (
	rngMu sync.Mutex
	rng   = rand.New(rand.NewSource(defaultSeed))
)

// defaultSeed is fixed so that an un-seeded process is still deterministic.
// Call Seed to choose a different stream.
const defaultSeed = 0x6E616E6F // "nano"

// Seed reseeds this backend's random source, making subsequent weight
// initialisation reproducible.
func (bk *CPUBackend) Seed(seed uint64) {
	rngMu.Lock()
	defer rngMu.Unlock()
	rng = rand.New(rand.NewSource(int64(seed)))
}

func (bk *CPUBackend) Normal(data []float32, mean, stdDev float32, size int) {
	rngMu.Lock()
	defer rngMu.Unlock()

	for i := 0; i < size; i++ {
		data[i] = mean + float32(rng.NormFloat64())*stdDev
	}
}
