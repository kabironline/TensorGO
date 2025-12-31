package tensor

// buildTopologicalOrder builds a topological order of tensors for backpropagation.
func buildTopologicalOrder(t *Tensor, visited map[*Tensor]bool) *[]*Tensor {
	if visited[t] {
		return nil
	}

	visited[t] = true
	var order []*Tensor

	for _, parent := range t.Parents {
		parentOrder := buildTopologicalOrder(parent, visited)
		if parentOrder != nil {
			order = append(order, *parentOrder...)
		}
	}

	order = append(order, t)
	return &order
}

// BackProp performs backpropagation to compute gradients for all tensors
// in the computation graph leading to tensor t.
func (t *Tensor) BackProp() {
	visited := make(map[*Tensor]bool)
	order := buildTopologicalOrder(t, visited)

	// Initialize the gradient of the output tensor to 1
	for i := range t.Grad {
		t.Grad[i] = 1.0
	}

	// Traverse the graph in reverse topological order
	for i := len(*order) - 1; i >= 0; i-- {
		current := (*order)[i]
		if current.Backward != nil {
			current.Backward()
		}
	}
}
