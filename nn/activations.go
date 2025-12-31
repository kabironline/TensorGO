package nn

import "github.com/kabironline/nanograd/tensor"

type ReLU struct{}

func (r *ReLU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.ReLU() }
func (r *ReLU) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }

type Sigmoid struct{}

func (s *Sigmoid) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Sigmoid() }
func (s *Sigmoid) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }

type Tanh struct{}

func (t *Tanh) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Tanh() }
func (t *Tanh) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }

type Softmax struct{}

func (t *Softmax) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softmax() }
func (t *Softmax) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{} }
