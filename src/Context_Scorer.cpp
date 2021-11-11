#include "Context_Scorer.hpp"

#include "Tokenizer.cpp"

class Vocinity::Context_Scorer::Scorer_Backend
{
  public:
    Scorer_Backend(const std::filesystem::__cxx11::path& scorer_model_path = std::string()
#ifdef CUDA_AVAILABLE
                       ,
                   const Inference_Backend device = Inference_Backend::CPU
#endif
    )
    {
        if(not scorer_model_path.empty())
        {
            _scorer_model = torch::jit::load(scorer_model_path.string());
#ifdef CUDA_AVAILABLE
            if(device == Inference_Backend::CUDA)
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
    }

    ~Scorer_Backend() = default;

  public:
    at::Tensor score(const at::Tensor& input_ids, const at::Tensor& att_mask);

    static constexpr ushort get_max_sequence_length()
    {
        return 1024;
    }

    static constexpr int64_t get_label_ignore_id()
    {
        return -100;
    }

    static constexpr int64_t get_stride()
    {
        return 512;
    }


  private:
    torch::jit::script::Module _scorer_model;
    std::mutex instanceMutex;
    c10::InferenceMode guard{true};
};

Vocinity::Context_Scorer::Context_Scorer(const std::filesystem::path& scorer_model_path,
                                         const Model_Family& family,
                                         const Tokenizer_Configuration& encoding_conf
#ifdef CUDA_AVAILABLE
                                         ,
                                         const Inference_Backend device
#endif
                                         )
    :
#ifdef CUDA_AVAILABLE
    _torch_runtime(std::make_unique<Scorer_Backend>(scorer_model_path, device))
    , _device(device == Inference_Backend::CUDA ? torch::kCUDA : torch::kCPU)
#else
    : _torch_runtime(std::make_unique<Scorer_Backend>(scorer_model_path))
#endif
    , _tokenizer(std::make_unique<Tokenizer>(encoding_conf.vocab_file,
                                             encoding_conf.merge_file,
                                             encoding_conf.bos_token_str,
                                             encoding_conf.eos_token_str,
                                             encoding_conf.pad_token_str,
                                             encoding_conf.unk_token_str,
                                             encoding_conf.mask_token_str))
    , _family(family)
{}

Vocinity::Context_Scorer::~Context_Scorer()
{}

void
Vocinity::Context_Scorer::optimize_parallelization_policy_for_use_of_multiple_instances()
{
    at::set_num_interop_threads(1);
    at::set_num_threads(1);
}

void
Vocinity::Context_Scorer::optimize_parallelization_policy_for_use_of_single_instance()
{
    static const ushort physical_cores = std::thread::hardware_concurrency() / 2;
    at::set_num_interop_threads(physical_cores);
    at::set_num_threads(physical_cores);
}

torch::Tensor
Vocinity::Context_Scorer::Scorer_Backend::score(const at::Tensor& input_ids,
                                                const at::Tensor& att_mask)
{
    const std::lock_guard<std::mutex> lock(instanceMutex);
    return _scorer_model
        .forward(std::vector<torch::jit::IValue>{torch::jit::IValue(input_ids),
                                                 torch::jit::IValue(att_mask)})
        .toTuple()
        ->elements()
        .at(0)
        .toTensor()
        .detach();
}

#include <torch/csrc/api/include/torch/nn/functional/loss.h>

Vocinity::Context_Scorer::Score
Vocinity::Context_Scorer::score(const std::string& sentence, const bool per_char_normalized)
{
    auto [input_ids, input_mask, actual_token_size] = encode(sentence);
    const unsigned long sequence_length             = input_ids.size(-1);
    if(_family == Model_Family::Neo)
    {
        input_ids  = input_ids.unsqueeze(0);
        input_mask = input_mask.unsqueeze(0);
    }

    std::vector<Score> total_score;
    for(size_t i = 0; i < sequence_length; i += _torch_runtime->get_stride())
    {
        const auto begin_loc = i;
        const auto end_loc =
            std::min(i + _torch_runtime->get_max_sequence_length(), sequence_length);
        const auto current_actual_token_end_loc =
            std::min(end_loc, std::max(actual_token_size + 2 - begin_loc, (unsigned long) 0));
        const auto trg_len = end_loc - i;
        const auto& current_input_ids =
            input_ids.index({Slice(begin_loc, end_loc)}).to(_device);
        const auto& current_att_mask =
            input_mask.index({Slice(begin_loc, end_loc)}).to(_device);
        auto target_ids = current_input_ids.clone();
        target_ids.index_put_({Slice(None, -trg_len)}, _torch_runtime->get_label_ignore_id());

        const auto& logits = _torch_runtime->score(current_input_ids, current_att_mask);

        const auto& loss      = process_labels(target_ids, logits);
        const auto& log_probs = torch::nn::functional::log_softmax(
            logits, torch::nn::functional::LogSoftmaxFuncOptions(-1));

        torch::Tensor target_log_probs;
        if(_family == Model_Family::OpenAI)
        {
            const auto& out_mask = input_ids.index({Slice(
                std::min(std::max((unsigned long) 1, begin_loc), current_actual_token_end_loc),
                current_actual_token_end_loc)});
            target_log_probs     = log_probs.gather(-1, out_mask.unsqueeze(-1)).squeeze(-1);
        }
        else
        {
            const auto out_mask =
                input_ids[0]
                    .index({Slice(std::min(std::max((unsigned long) 1, begin_loc),
                                           current_actual_token_end_loc),
                                  current_actual_token_end_loc)})
                    .unsqueeze(0)
                    .unsqueeze(-1);
            target_log_probs = log_probs.gather(-1, out_mask).squeeze(-1).squeeze(0);
        }

        Score score;

        const auto neg_log_likelihood = loss * torch::Scalar(int64_t(trg_len));
        score.negative_log_likelihood = neg_log_likelihood.item().toDouble();

        const auto prod_score = target_log_probs.sum();
        score.production      = prod_score.item().toDouble();

        const auto mean_score =
            current_actual_token_end_loc > 0
                ? target_log_probs.logsumexp(0) - std::log(current_actual_token_end_loc - 1)
                : torch::tensor(0).to(_device);
        score.mean = mean_score.exp().item().toDouble();

        const auto gmean_score = target_log_probs.mean(0);
        score.g_mean           = gmean_score.exp().item().toDouble();

        const auto hmean_score = current_actual_token_end_loc > 0
                                     ? target_log_probs.neg().logsumexp(0).neg()
                                           + std::log(current_actual_token_end_loc - 1)
                                     : torch::tensor(0).to(_device);
        score.h_mean           = hmean_score.item().toDouble();

        score.loss = loss.exp().item().toDouble();

        score.sentence_probability = current_actual_token_end_loc > 0
                                         ? -1
                                               * std::exp(-1 * loss.item().toDouble()
                                                          * (current_actual_token_end_loc - 1))
                                         : 0;
        total_score.emplace_back(std::move(score));

        if(end_loc == sequence_length)
        {
            break;
        }
    }

    Score score;
    for(ushort chunk_result_order = 0; chunk_result_order < total_score.size();
        ++chunk_result_order)
    {
        score.negative_log_likelihood +=
            total_score.at(chunk_result_order).negative_log_likelihood;
        score.production += total_score.at(chunk_result_order).production;
        score.mean += total_score.at(chunk_result_order).mean;
        score.g_mean += total_score.at(chunk_result_order).g_mean;
        score.h_mean += total_score.at(chunk_result_order).h_mean;
        score.loss += total_score.at(chunk_result_order).loss;
        score.sentence_probability += total_score.at(chunk_result_order).sentence_probability;
    }

    if(per_char_normalized)
    {
        score.negative_log_likelihood /= (actual_token_size + 1);
        score.production /= (actual_token_size + 1);
        score.mean /= (actual_token_size + 1);
        score.g_mean /= (actual_token_size + 1);
        score.h_mean /= (actual_token_size + 1);
        score.loss /= (actual_token_size + 1);
        score.sentence_probability /= (actual_token_size + 1);
    }

    return score;
}

torch::Tensor
Vocinity::Context_Scorer::process_labels(const torch::Tensor& labels,
                                         const torch::Tensor& logits)
{
    torch::Tensor loss;
    if(labels.numel())
    {
        const auto& shift_logits =
            logits.index({"...", Slice(None, -1), Slice()}).contiguous();
        const auto& shift_labels = labels.index({"...", Slice(1, None)}).contiguous();
        loss                     = torch::nn::functional::cross_entropy(
            shift_logits.view({-1, shift_logits.size(-1)}), shift_labels.view({-1}));
    }
    return loss;
}

Vocinity::Context_Scorer::Encoded_Sequence
Vocinity::Context_Scorer::encode(const std::string& sentence, const bool parallel)
{
    const auto tokens = _tokenizer->tokenize(sentence);
    std::vector<int64_t> ids;
    ids.resize(tokens.size() + 2);
    if(parallel)
    {
#ifdef CPP17_AVAILABLE
        std::transform(std::execution::par_unseq,
                       tokens.begin(),
                       tokens.end(),
                       ids.begin() + 1,
                       [this](const auto& token) -> int64_t
                       { return _tokenizer->convert_token_to_id(token); }); // moves
#else
        __gnu_parallel::transform(tokens.begin(),
                                  tokens.end(),
                                  ids.begin() + 1,
                                  [this](const auto& token) -> int64_t
                                  { return _tokenizer->convert_token_to_id(token); }); // moves
#endif
    }
    else
    {
        std::transform(tokens.begin(),
                       tokens.end(),
                       ids.begin() + 1,
                       [this](const auto& token)
                       { return _tokenizer->convert_token_to_id(token); }); // moves
    }
    const Actual_Token_Size actual_token_size = ids.size() - 2;

    ids[0]                    = _tokenizer->get_bos_token_id();
    ids[ids.size() - 1]       = _tokenizer->get_eos_token_id();
    const auto& tokens_padded = akil::memory::vector_1d_to_tensor_1d_no_copy<int64_t>(ids);
    const uint64_t padded_token_size = ids.size();

    const uint64_t full_sequence_size =
        (padded_token_size % _torch_runtime->get_max_sequence_length())
            ? _torch_runtime->get_max_sequence_length()
                  - (padded_token_size % _torch_runtime->get_max_sequence_length())
                  + padded_token_size
            : padded_token_size;
    auto full_sequence = torch::full(full_sequence_size,
                                     _tokenizer->get_pad_token_id(),
                                     torch::TensorOptions().dtype(torch::kInt64));
    full_sequence.index_put_({Slice(None, padded_token_size)}, tokens_padded);


    auto input_mask = torch::zeros(full_sequence_size);
    input_mask.index_put_({Slice(None, padded_token_size)}, 1);

    return {full_sequence.to(_device), input_mask.to(_device), actual_token_size};
}
