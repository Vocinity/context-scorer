#ifndef ABSTRACT_SCORER_BACKEND_HPP
#define ABSTRACT_SCORER_BACKEND_HPP

#include "../Context-Scorer.hpp"

class Vocinity::Context_Scorer::Scorer_Backend
{
  public:
    struct GPT_Configuration
    {
        ushort hidden_size         = 768; //n_embd
        ushort num_attention_heads = 12;  // n_head
        ushort num_layers          = 6;   //n_layer (hidden_layers)
        ushort sequence_length=1024; // n_ctx
    };

    struct Inference_Output
    {
        torch::Tensor loss;
        torch::Tensor logits;
        torch::Tensor present;
    };

  public:
    static inline GPT_Configuration get_configuration(
        const Vocinity::Context_Scorer::GPT_TYPE type)
    {
        if(type == Vocinity::Context_Scorer::GPT_TYPE::DistilGPT2)
        {
            return GPT_Configuration{768, 12, 6,1024};
        }
        return GPT_Configuration();
    }

    static inline bool is_neo_like_structure(const Vocinity::Context_Scorer::GPT_TYPE type)
    {
        return magic_enum::enum_integer(type) > 8;
    }

    static inline c10::Device get_torch_device(
        const Vocinity::Context_Scorer::Inference_Backend& backend)
    {
        if(backend == Vocinity::Context_Scorer::Inference_Backend::CUDA)
        {
            return torch::kCUDA;
        }
        return torch::kCPU;
    }

    c10::ScalarType get_input_int_range() const
    {
        return _input_int_range;
    }

  public:
    virtual ~Scorer_Backend() = default;

  public:
    virtual Inference_Output score(const at::Tensor& input_ids,
                                   const at::Tensor& att_mask,
                                   const torch::Tensor& labels = torch::Tensor(),
                                   const torch::Tensor& past   = torch::Tensor()) = 0;

    virtual ushort get_max_sequence_length() = 0;

    virtual int64_t get_label_ignore_id() = 0;

    virtual int64_t get_stride() = 0;

  protected:
    Vocinity::Context_Scorer::Inference_Backend _device =
        Vocinity::Context_Scorer::Inference_Backend::CPU;
    const c10::ScalarType _input_int_range=torch::kInt64;
};


#endif
