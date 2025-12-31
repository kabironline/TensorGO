package nn

import "github.com/kabironline/nanograd/tensor"

type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
}
