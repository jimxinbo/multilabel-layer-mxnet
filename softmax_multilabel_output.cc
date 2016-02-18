/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_multilabel_output.cc
 * \brief
 * modified from softmax_output.cc by Bo Xin
*/
#include "./bx_softmax_multilabel_output-inl.h"


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxMultilabelOutputParam param) {
  return new SoftmaxMultilabelOutputOp<cpu>(param);
}

Operator *SoftmaxMultilabelOutputProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SoftmaxMultilabelOutputParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxMultilabelOutput, SoftmaxMultilabelOutputProp)
.describe("Perform a softmax_multilabel transformation on input, backprop with logloss.")
.add_argument("data", "Symbol", "Input data to softmax_multilabel.")
.add_arguments(SoftmaxMultilabelOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet



