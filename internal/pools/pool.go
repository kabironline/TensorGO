package pools

import (
	"fmt"
	"math/bits"
	"sync"
)

// Define buckets for powers of 2: 2^6 (64) to 2^28 (256MB)
var pools [29]*sync.Pool

func init() {
	for i := 6; i < 29; i++ {
		size := 1 << i // Calculate 2^i
		pools[i] = &sync.Pool{
			New: func() any {
				return make([]float64, 0, size)
			},
		}
	}
}

func GetBuffer(size int) []float64 {
	// 1. Find the bucket index (Log2 logic)
	// You can use bits.Len(uint(size)) from "math/bits" for speed
	bucket := bits.Len(uint(size - 1))
	bucket = max(bucket, 6)
	if bucket >= 29 {
		fmt.Printf("Too big buffer requested: %d elements\n", size)
		return make([]float64, size)
	} // Too big for pool

	// 2. Get from specific pool
	ptr := pools[bucket].Get().([]float64)

	// 3. Resize capacity if needed (rare, but safety check)
	if cap(ptr) < size {
		return make([]float64, size)
	}

	// 4. Set length
	return ptr[:size]
}

func GetZeroedBuffer(size int) []float64 {
	buf := GetBuffer(size)
	// Zero only the used length portion:
	for i := range buf {
		buf[i] = 0.0
	}
	return buf
}

func PutBuffer(buf []float64) {
	bucket := bits.Len(uint(cap(buf) - 1))
	if bucket >= 6 && bucket < 29 {
		pools[bucket].Put(buf)
	}
}
