/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_multilabel_output-inl.h
 * \brief
 * modified from softmax_output-inl.h by Bo Xin
*/
#ifndef MXNET_OPERATOR_SOFTMAX_MULTILABEL_OUTPUT_INL_H_
#define MXNET_OPERATOR_SOFTMAX_MULTILABEL_OUTPUT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace softmaxmultilabelout_enum {
enum SoftmaxMultilabelOutputOpInputs {kData, kLabel};
enum SoftmaxMultilabelOutputOpOutputs {kOut};
}  // namespace softmaxmultilabelout_enum

struct SoftmaxMultilabelOutputParam : public dmlc::Parameter<SoftmaxMultilabelOutputParam> {
  float grad_scale;
  int num_label;
  DMLC_DECLARE_PARAMETER(SoftmaxMultilabelOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
	DMLC_DECLARE_FIELD(num_label).set_default(1)
		.describe("Set the number of labels");
  };
};

template<typename xpu>
class SoftmaxMultilabelOutputOp : public Operator {
 public:
  explicit SoftmaxMultilabelOutputOp(SoftmaxMultilabelOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "SoftmaxMultilabelOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "SoftmaxMultilabelOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[softmaxmultilabelout_enum::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[softmaxmultilabelout_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Softmax(out, data);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 2> label = in_data[softmaxmultilabelout_enum::kLabel].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[softmaxmultilabelout_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad = in_grad[softmaxmultilabelout_enum::kData].FlatTo2D<xpu, real_t>(s);
    SoftmaxMultilabelGrad(grad, out, label);
    if (param_.grad_scale < 1.0) {
      grad *= param_.grad_scale;  
    }
  }

 private:
  SoftmaxMultilabelOutputParam param_;
};  // class SoftmaxMultilabelOutputOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxMultilabelOutputParam param);

#if DMLC_USE_CXX11
class SoftmaxMultilabelOutputProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
	SHAPE_ASSIGN_CHECK(*in_shape, softmaxmultilabelout_enum::kLabel, Shape2(dshape[0], param_.num_label));
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
	  auto ptr = new SoftmaxMultilabelOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SoftmaxMultilabelOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[softmaxmultilabelout_enum::kLabel], out_data[softmaxmultilabelout_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[softmaxmultilabelout_enum::kOut], in_grad[softmaxmultilabelout_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[softmaxmultilabelout_enum::kData], out_data[softmaxmultilabelout_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 protected:
	 SoftmaxMultilabelOutputParam param_;
};  // class SoftmaxMultilabelOutputProp


#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet




// detailed implementaion of both cpu and gpu
namespace mshadow {

	template<typename DType>
	inline void SoftmaxMultilabelGrad(Tensor<cpu, 2, DType> dst,
		const Tensor<cpu, 2, DType> &src,
		const Tensor<cpu, 2, DType> &label) {

		for (index_t y = 0; y < dst.size(0); ++y) {
			for (index_t x = 0; x < dst.size(1); ++x) {
				dst[y][x] = 0.0;
				for (index_t i = 0; i < label.size(1); ++i) {
					const index_t k = static_cast<int>(label[y][i]);
					
					if ( k >= 0 && k < dst.size(1)) {

						if (x == k) {
							dst[y][x] += src[y][x] - 1.0f;
						}
						else {
							dst[y][x] += src[y][x];
						}

					}

				}
			}
		}
	}


	namespace cuda {
		template<int x_bits, typename DType, typename DstPlan, typename SrcPlan1, typename SrcPlan2>
		__global__ void SoftmaxMultilabelGradKernel(DstPlan dst, SrcPlan1 src, SrcPlan2 label, index_t xmax, index_t lmax) {
			const unsigned x_size = 1 << x_bits;
			const int y = blockIdx.x;

			// calculate normalizer, with writeback
			for (unsigned x = 0; x < xmax; x += x_size) {
				const unsigned xindex = x + threadIdx.x;
				if (xindex < xmax) {

					dst.REval(y, xindex) = 0.0f;

					for (unsigned i = 0; i < lmax; ++i) {

						//__syncthreads();

						int k = static_cast<int>(label.Eval(y, i));
							
						if (k >= 0 && k < xmax) {

							if (xindex == k) {
								dst.REval(y, xindex) += src.Eval(y, xindex) - 1.0f;
							}
							else {
								dst.REval(y, xindex) += src.Eval(y, xindex);
							}

						}

					}
				}
			}
		}
		template<typename DType>
		inline void SoftmaxMultilabelGrad(Tensor<gpu, 2, DType> &dst,
			const Tensor<gpu, 2, DType> &src,
			const Tensor<gpu, 2, DType> &label) {
			dim3 dimBlock(kBaseThreadNum);
			dim3 dimGrid(dst.size(0));
			CHECK_EQ(dst.shape_, src.shape_) << "SoftmaxMultilabelGrad: shape mismatch";
			CHECK_EQ(dst.size(0), label.size(0)) << "SoftmaxMultilabelGrad: label shape mismatch";
			CheckLaunchParam(dimGrid, dimBlock, "SoftmaxMultilabelGrad");
			cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
			SoftmaxMultilabelGradKernel<kBaseThreadBits, DType>
				<< <dimGrid, dimBlock, 0, stream >> >
				(expr::MakePlan(dst),
				expr::MakePlan(src),
				expr::MakePlan(label),
				dst.size(1), label.size(1));
		}
	} // namespace cuda

	template<typename DType>
	inline void SoftmaxMultilabelGrad(Tensor<gpu, 2, DType> dst,
		const Tensor<gpu, 2, DType> &src,
		const Tensor<gpu, 2, DType> &label) {
		cuda::SoftmaxMultilabelGrad(dst, src, label);
	}

} // namespace mshadow


#endif  // MXNET_OPERATOR_SOFTMAX_MULTILABEL_OUTPUT_INL_H_
