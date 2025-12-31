package optim

type Optimizer interface {
	Step()     // Updates the weights
	ZeroGrad() // Resets all gradients to zero
}
