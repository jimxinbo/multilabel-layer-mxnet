/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_multilabel_output.cu
 * \brief
 * modified from softmax_output.cu by Bo Xin
*/

#include "./softmax_multilabel_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SoftmaxMultilabelOutputParam param) {
  return new SoftmaxMultilabelOutputOp<gpu>(param);
}


}  // namespace op
}  // namespace mxnet

