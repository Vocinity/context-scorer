#ifndef TORCH_SCORER_BACKEND_HPP
#define TORCH_SCORER_BACKEND_HPP

#include "Abstract-Scorer-Backend.hpp"

#include <torch/csrc/api/include/torch/nn/functional/loss.h>

class Scorer_Torch_Backend : public Vocinity::Context_Scorer::Abstract_Scorer_Backend
{
  public:
    Scorer_Torch_Backend(
        const std::filesystem::__cxx11::path& scorer_model_path = std::string(),
        const Vocinity::Context_Scorer::Precision precision =
            Vocinity::Context_Scorer::Precision::FP32,
        const Vocinity::Context_Scorer::GPT_TYPE type =
            Vocinity::Context_Scorer::GPT_TYPE::DistilGPT2
#ifdef CUDA_AVAILABLE
        ,
        const Vocinity::Context_Scorer::Inference_Hardware device =
            Vocinity::Context_Scorer::Inference_Hardware::CPU
#endif
    )
    {
        _precision = precision;
        if(scorer_model_path.empty())
        {
            throw std::runtime_error("scorer_model_path can not be empty");
        }
        _scorer_model = torch::jit::load(scorer_model_path.string());
#ifdef CUDA_AVAILABLE
        if(device == Vocinity::Context_Scorer::Inference_Hardware::CUDA)
        {
            _scorer_model.to(torch::kCUDA);
        }
        else
#endif
        {
            _scorer_model.to(torch::kCPU);
        }

        _scorer_model.eval();
    }

    ~Scorer_Torch_Backend() override = default;

  public:
    Inference_Output score(const at::Tensor& input_ids,
                           const at::Tensor& att_mask,
                           const torch::Tensor& labels,
                           const torch::Tensor& past) override
    {
        const std::lock_guard<std::mutex> lock(instanceMutex);

        const auto logits =
            _scorer_model
                .forward(std::vector<torch::jit::IValue>{torch::jit::IValue(input_ids),
                                                         torch::jit::IValue(att_mask)})
                .toTuple()
                ->elements()
                .at(0)
                .toTensor()
                .detach();

        const auto& shift_logits =
            logits.index({"...", Slice(None, -1), Slice()}).contiguous();
        const auto& shift_labels = labels.index({"...", Slice(1, None)}).contiguous();
        const auto loss          = torch::nn::functional::cross_entropy(
            shift_logits.view({-1, shift_logits.size(-1)}), shift_labels.view({-1}));
        return {loss, logits, past};
    }

    virtual ushort get_max_sequence_length() override
    {
        return std::max(_max_input_sequence_length / 16, (int64_t) 64);
    }

    virtual int64_t get_label_ignore_id() override
    {
        return -100;
    }

    virtual int64_t get_stride() override
    {
        return get_max_sequence_length() / 2;
    }

    virtual c10::ScalarType get_torch_precision() const override
    {
        c10::ScalarType precision;
#ifdef CUDA_FP16_AVAILABLE
        if(_precision == Vocinity::Context_Scorer::Precision::FP16)
        {
            precision = torch::kFloat16;
        }
        else if(_precision == Vocinity::Context_Scorer::Precision::FP32)
#endif
        {
            precision = torch::kFloat32;
        }
        return precision;
    }

  private:
    torch::jit::script::Module _scorer_model;
    std::mutex instanceMutex;
    c10::InferenceMode guard{true};
    static inline constexpr int64_t _max_input_sequence_length = 1024;
};

#endif
