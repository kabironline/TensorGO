package tensor

import "fmt"

// buildTopologicalOrder builds a topological order of tensors for backpropagation.
// buildTopologicalOrder builds a post-order (parents before child) topological
// ordering for backprop without recursive slice allocations. It returns a
// slice of tensors where parents appear before their dependents.
func buildTopologicalOrder(root *Tensor, visited map[*Tensor]bool) []*Tensor {
	type frame struct {
		node      *Tensor
		processed bool
	}

	order := make([]*Tensor, 0)
	stack := make([]frame, 0, 64)
	stack = append(stack, frame{node: root, processed: false})

	for len(stack) > 0 {
		fr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if fr.processed {
			order = append(order, fr.node)
			continue
		}

		if visited[fr.node] {
			continue
		}

		// Mark visited to avoid re-pushing the same node's parents.
		visited[fr.node] = true

		// Post-order: push the node marked as processed, then its parents.
		stack = append(stack, frame{node: fr.node, processed: true})
		for _, p := range fr.node.Parents {
			if !visited[p] {
				stack = append(stack, frame{node: p, processed: false})
			}
		}
	}

	return order
}

// BackProp performs backpropagation to compute gradients for all tensors
// in the computation graph leading to tensor t.
func (t *Tensor) BackProp() {
	// A gradient is only defined for a scalar output. Seeding a non-scalar root
	// with all-ones silently computes the gradient of sum(t) instead, which is a
	// different function -- say so rather than returning plausible wrong numbers.
	if n := TotalSize(t.Shape); n != 1 {
		panic(fmt.Sprintf(
			"BackProp: gradients can only be created implicitly for a scalar output, "+
				"but this tensor has shape %v (%d elements); reduce it first (.Sum()) "+
				"or use BackPropWith to supply an explicit seed gradient", t.Shape, n))
	}

	seed := t.Device.Allocate(1)
	defer t.Device.Free(seed)
	t.Device.Fill(seed, 1.0, 1)

	t.BackPropWith(seed)
}

// BackPropWith runs backpropagation from this tensor using an explicit seed
// gradient -- the equivalent of PyTorch's backward(gradient=...).
//
// seed must be in logical order and TotalSize(t.Shape) elements long. Use this
// for a non-scalar root, or to weight the output before differentiating.
func (t *Tensor) BackPropWith(seed []float32) {
	n := TotalSize(t.Shape)
	if len(seed) != n {
		panic(fmt.Sprintf("BackPropWith: seed has %d elements, want %d for shape %v",
			len(seed), n, t.Shape))
	}

	visited := make(map[*Tensor]bool)
	order := buildTopologicalOrder(t, visited)

	t.ensureGrad()
	t.Device.Copy(t.Grad(), seed)

	// Traverse the graph in reverse topological order.
	for i := len(order) - 1; i >= 0; i-- {
		current := order[i]
		if current.Backward != nil && current.Grad() != nil {
			current.Backward()
		}
	}
}
