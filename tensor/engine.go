package tensor

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
	visited := make(map[*Tensor]bool)
	order := buildTopologicalOrder(t, visited)

	// Ensure root tensor has a gradient buffer and initialize it to 1
	t.ensureGrad()
	t.Device.Fill(t.Grad(), 1.0, len(t.Grad()))

	// Traverse the graph in reverse topological order
	for i := len(order) - 1; i >= 0; i-- {
		current := order[i]
		if current.Backward != nil && current.Grad() != nil {
			current.Backward()
		}
	}
}
