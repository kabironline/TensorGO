package cpu

import (
	"runtime"
	"sync"
)

// Task represents a function that processes a chunk of data.
type Task func(start, end int)

// WorkerPool manages a fixed set of goroutines for parallel processing.
// This is primarily used for CPU-bound tensor operations.
type WorkerPool struct {
	numWorkers int
	tasks      chan taskWrapper
	wg         sync.WaitGroup
	quit       chan struct{}
}

type taskWrapper struct {
	task  Task
	start int
	end   int
}

// NewWorkerPool creates and starts a new worker pool.
func NewWorkerPool(numWorkers int) *WorkerPool {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	p := &WorkerPool{
		numWorkers: numWorkers,
		tasks:      make(chan taskWrapper, numWorkers*2),
		quit:       make(chan struct{}),
	}

	p.start()
	return p
}

func (p *WorkerPool) start() {
	for i := 0; i < p.numWorkers; i++ {
		go func() {
			for {
				select {
				case t := <-p.tasks:
					t.task(t.start, t.end)
					p.wg.Done()
				case <-p.quit:
					return
				}
			}
		}()
	}
}

// Process parallelizes a task across the worker pool for a given total size.
// It splits the work into chunks based on the number of workers.
func (p *WorkerPool) Process(totalSize int, task Task) {
	if totalSize <= 0 {
		return
	}

	chunkSize := (totalSize + p.numWorkers - 1) / p.numWorkers
	for i := 0; i < p.numWorkers; i++ {
		start := i * chunkSize
		if start >= totalSize {
			break
		}
		end := start + chunkSize
		if end > totalSize {
			end = totalSize
		}

		p.wg.Add(1)
		p.tasks <- taskWrapper{
			task:  task,
			start: start,
			end:   end,
		}
	}
	p.wg.Wait()
}

// Close stops all workers in the pool.
func (p *WorkerPool) Close() {
	close(p.quit)
}
