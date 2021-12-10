#ifndef ABSTRACT_SCORER_BACKEND_HPP
#define ABSTRACT_SCORER_BACKEND_HPP

#include "Context-Scorer.hpp"

class Vocinity::Context_Scorer::Scorer_Backend
{
  public:
    virtual ~Scorer_Backend() = default;

  public:
    virtual std::pair<torch::Tensor, torch::Tensor> score(
        const at::Tensor& input_ids,
        const at::Tensor& att_mask,
        const torch::Tensor& labels = torch::Tensor()) = 0;

    virtual ushort get_max_sequence_length() = 0;

    virtual int64_t get_label_ignore_id() = 0;

    virtual int64_t get_stride() = 0;
};


#endif
