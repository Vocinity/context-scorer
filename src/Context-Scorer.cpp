#include "Tokenizer.cpp"
#include "backends/ONNX-Scorer-Backend.hpp"
#include "backends/Torch-Scorer-Backend.hpp"

#include <torch/csrc/api/include/torch/nn/functional/padding.h>

class Context_Scorer_Model_Resource_Manager
{
  public:
    Context_Scorer_Model_Resource_Manager()
    {
        Q_INIT_RESOURCE(context_scorer_bag);
    }
    ~Context_Scorer_Model_Resource_Manager()
    {
        Q_CLEANUP_RESOURCE(context_scorer_bag);
    }
};

Vocinity::Context_Scorer::Context_Scorer(const std::filesystem::path& scorer_model_path,
                                         const GPT_TYPE type,
                                         const Tokenizer_Configuration& encoding_conf,
                                         const Precision precision
#ifdef CUDA_AVAILABLE
                                         ,
                                         const Inference_Environment environment
#endif
                                         )
    : _precision(precision)
    , _type(type)
#ifdef CUDA_AVAILABLE
    , _torch_device((environment == Inference_Environment::CUDA
                    or environment == Inference_Environment::TensorRT)
                       ? torch::kCUDA
                       : torch::kCPU)
#endif
    , _tokenizer(std::make_unique<Tokenizer>(encoding_conf.vocab_file,
                                             encoding_conf.merge_file,
                                             encoding_conf.bos_token_str,
                                             encoding_conf.eos_token_str,
                                             encoding_conf.pad_token_str,
                                             encoding_conf.unk_token_str,
                                             encoding_conf.mask_token_str))
{
    {
        const std::lock_guard lock(_instance_mutex);
        static Context_Scorer_Model_Resource_Manager resources;
    }

#ifdef CUDA_AVAILABLE
    if(environment == Inference_Environment::CUDA
       or environment == Inference_Environment::TensorRT)
    {
#	ifdef ONNX_AVAILABLE
        _inference_backend = std::make_unique<Scorer_ONNX_Backend>(
            scorer_model_path, precision, _tokenizer->get_vocab_size(), type, environment);
#	else
        _inference_backend = std::make_unique<Scorer_Torch_Backend>(
            scorer_model_path, precision, type, environment);
#	endif
    }
    else
#endif
    {
#ifdef ONNX_AVAILABLE
        _inference_backend = std::make_unique<Scorer_ONNX_Backend>(
            scorer_model_path, precision, _tokenizer->get_vocab_size(), type);
#else
        _inference_backend =
            std::make_unique<Scorer_Torch_Backend>(scorer_model_path, precision, type);
#endif
    }
}

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

Vocinity::Context_Scorer::Score
Vocinity::Context_Scorer::score_context(const std::string& sentence,
                                        const bool per_char_normalized,
                                        const bool intra_batching)
{
    const std::lock_guard lock(_instance_mutex);
    const auto& encoding = encode(sentence
#ifndef CPP17_AVAILABLE
                                  ,
                                  sentence.size() >= std::thread::hardware_concurrency() * 1024
#endif
    );
    Score score;
    if(intra_batching
       and (encoding.input_ids.size(-1) > _inference_backend->get_max_sequence_length()))
    {
        score = score_long_context(encoding, per_char_normalized);
    }
    else
    {
        score = score_short_context(encoding, per_char_normalized);
    }
    score.utterance = sentence;
    return score;
}

Vocinity::Context_Scorer::Score
Vocinity::Context_Scorer::score_short_context(
    const Vocinity::Context_Scorer::Encoded_Sequence& encoding,
    const bool per_char_normalized)
{
    const auto [input_ids, input_mask, actual_token_size] = encoding;
    const unsigned long actual_sequence_length            = input_ids.size(-1);

    const torch::Tensor past = _inference_backend->get_initial_single_batch_past();

    std::vector<Score> total_score;
    for(size_t i = 0; i < actual_sequence_length; i += _inference_backend->get_stride())
    {
        const auto begin_loc = i;
        const auto end_loc   = std::min(i + _inference_backend->get_max_sequence_length(),
                                      actual_sequence_length);
        const auto current_actual_token_end_loc =
            std::min(end_loc, std::max(actual_token_size + 2 - begin_loc, (unsigned long) 0));
        const auto trg_len     = end_loc - begin_loc;
        auto current_input_ids = input_ids.index({Slice(begin_loc, end_loc)});
        auto current_att_mask  = input_mask.index({Slice(begin_loc, end_loc)});
        auto target_ids        = current_input_ids.clone();
        target_ids.index_put_({Slice(None, -trg_len)},
                              _inference_backend->get_label_ignore_id());

#ifndef ONNX_AVAILABLE
        if(Abstract_Scorer_Backend::is_neo_like_structure(_type))
#endif
        {
            current_input_ids = current_input_ids.unsqueeze(0);
            current_att_mask  = current_att_mask.unsqueeze(0);
            target_ids        = target_ids.unsqueeze(0);
        }

#ifdef CUDA_AVAILABLE
        if(_flush_torch_cuda_cache)
        {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
#endif

        Abstract_Scorer_Backend::Inference_Output payload =
            _inference_backend->score(current_input_ids, current_att_mask, target_ids, past);
#ifdef ONNX_AVAILABLE
        payload.logits = payload.logits.squeeze(0);
        payload.loss   = payload.loss.squeeze(0);
#endif

#ifdef ONNX_AVAILABLE
        //past = payload.present;
#endif

        Score score;
        if(payload.logits.numel())
        {
            const auto log_probs = torch::nn::functional::log_softmax(
                payload.logits, torch::nn::functional::LogSoftmaxFuncOptions(-1));

            torch::Tensor target_log_probs;
            if(not Abstract_Scorer_Backend::is_neo_like_structure(_type))
            {
                const auto& out_mask =
                    input_ids.index({Slice(std::min(std::max((unsigned long) 1, begin_loc),
                                                    current_actual_token_end_loc),
                                           current_actual_token_end_loc)});
                target_log_probs = log_probs.gather(-1, out_mask.unsqueeze(-1)).squeeze(-1);
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

            const auto prod_score  = target_log_probs.sum();
            const auto mean_score  = current_actual_token_end_loc > 0
                                         ? target_log_probs.logsumexp(0)
                                              - std::log(current_actual_token_end_loc - 1)
                                         : torch::tensor(0);
            const auto gmean_score = target_log_probs.mean(0);
            const auto hmean_score = current_actual_token_end_loc > 0
                                         ? target_log_probs.neg().logsumexp(0).neg()
                                               + std::log(current_actual_token_end_loc - 1)
                                         : torch::tensor(0);

            score.production = prod_score.item().toDouble();

            const auto mean_exp = mean_score.exp();
            if(not torch::isnan(mean_exp).any().item<bool>()
               and not torch::isinf(mean_exp).any().item<bool>())
            {
                score.mean = mean_exp.item().toDouble();
            }

            const auto g_mean_exp = gmean_score.exp();
            if(not torch::isnan(g_mean_exp).any().item<bool>()
               and not torch::isinf(g_mean_exp).any().item<bool>())
            {
                score.g_mean = g_mean_exp.item().toDouble();
            }

            if(not torch::isnan(hmean_score).any().item<bool>()
               and not torch::isinf(hmean_score).any().item<bool>())
            {
                score.h_mean = hmean_score.item().toDouble();
            }
        }

        if(payload.loss.numel())
        {
            const auto neg_log_likelihood = payload.loss * torch::Scalar(int64_t(trg_len));

            score.negative_log_likelihood = neg_log_likelihood.item().toDouble();

            const auto loss_exp = payload.loss.exp();
            if(not torch::isnan(loss_exp).any().item<bool>()
               and not torch::isinf(loss_exp).any().item<bool>())
            {
                score.loss = loss_exp.item().toDouble();
            }

            score.sentence_probability =
                current_actual_token_end_loc > 0
                    ? -1
                          * std::exp(-1 * payload.loss.item().toDouble()
                                     * (current_actual_token_end_loc - 1))
                    : 0;
        }

        total_score.emplace_back(std::move(score));

        if(end_loc == actual_sequence_length)
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

Vocinity::Context_Scorer::Score
Vocinity::Context_Scorer::score_long_context(
    const Vocinity::Context_Scorer::Encoded_Sequence& encoding,
    const bool per_char_normalized)
{
    const auto [input_ids, input_mask, actual_token_size] = encoding;
    const unsigned long actual_sequence_length            = input_ids.size(-1);

    const torch::Tensor past = _inference_backend->get_initial_single_batch_past();

    struct Batch_Metadata
    {
        ulong begin_loc                    = 0;
        ulong end_loc                      = 0;
        ulong current_actual_token_end_loc = 0;
        ulong trg_len                      = 0;
    };

    std::vector<Batch_Metadata> batch_metadata;
    torch::Tensor batched_input_ids, batched_att_masks, batched_labels, batched_pasts;
    for(size_t i = 0; i < actual_sequence_length; i += _inference_backend->get_stride())
    {
        const auto begin_loc = i;
        const auto end_loc   = std::min(i + _inference_backend->get_max_sequence_length(),
                                      actual_sequence_length);
        const auto current_actual_token_end_loc =
            std::min(end_loc, std::max(actual_token_size + 2 - begin_loc, (unsigned long) 0));
        const auto trg_len           = end_loc - begin_loc;
        const auto current_input_ids = input_ids.index({Slice(begin_loc, end_loc)});
        const auto current_att_mask  = input_mask.index({Slice(begin_loc, end_loc)});
        auto target_ids              = current_input_ids.clone();
        target_ids.index_put_({Slice(None, -trg_len)},
                              _inference_backend->get_label_ignore_id());

        if(batched_input_ids.numel())
        {
            batched_input_ids =
                torch::cat({batched_input_ids, current_input_ids.unsqueeze(0)}, 0);
            batched_att_masks =
                torch::cat({batched_att_masks, current_att_mask.unsqueeze(0)}, 0);
            batched_labels = torch::cat({batched_labels, target_ids.unsqueeze(0)}, 0);
#ifdef ONNX_AVAILABLE
            batched_pasts = torch::cat({batched_pasts, past}, 2);
#endif
        }
        else
        {
            batched_input_ids = current_input_ids.unsqueeze(0);
            batched_att_masks = current_att_mask.unsqueeze(0);
            batched_labels    = target_ids.unsqueeze(0);
            batched_pasts     = past;
        }

        batch_metadata.push_back({begin_loc, end_loc, current_actual_token_end_loc, trg_len});

        if(end_loc == actual_sequence_length)
        {
            break;
        }
    }

#ifdef CUDA_AVAILABLE
    if(_flush_torch_cuda_cache)
    {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif

    std::vector<Score> total_score;
    const auto batch_payload = _inference_backend->score(
        batched_input_ids, batched_att_masks, batched_labels, batched_pasts);
    for(uint batch_order = 0; batch_order < batch_metadata.size(); ++batch_order)
    {
        Score score;

        const auto& metadata = batch_metadata.at(batch_order);
        if(batch_payload.logits.numel())
        {
            const auto& logits = batch_payload.logits[batch_order];

            const auto log_probs = torch::nn::functional::log_softmax(
                logits, torch::nn::functional::LogSoftmaxFuncOptions(-1));
            const auto& out_mask = input_ids.index(
                {Slice(std::min(std::max((unsigned long) 1, metadata.begin_loc),
                                metadata.current_actual_token_end_loc),
                       metadata.current_actual_token_end_loc)});

            torch::Tensor target_log_probs =
                log_probs.gather(-1, out_mask.unsqueeze(-1)).squeeze(-1);

            const auto prod_score = target_log_probs.sum();
            const auto mean_score =
                metadata.current_actual_token_end_loc > 0
                    ? target_log_probs.logsumexp(0)
                          - std::log(metadata.current_actual_token_end_loc - 1)
                    : torch::tensor(0);
            const auto gmean_score = target_log_probs.mean(0);
            const auto hmean_score =
                metadata.current_actual_token_end_loc > 0
                    ? target_log_probs.neg().logsumexp(0).neg()
                          + std::log(metadata.current_actual_token_end_loc - 1)
                    : torch::tensor(0);

            score.production = prod_score.item().toDouble();

            const auto mean_exp = mean_score.exp();
            if(not torch::isnan(mean_exp).any().item<bool>()
               and not torch::isinf(mean_exp).any().item<bool>())
            {
                score.mean = mean_exp.item().toDouble();
            }

            const auto g_mean_exp = gmean_score.exp();
            if(not torch::isnan(g_mean_exp).any().item<bool>()
               and not torch::isinf(g_mean_exp).any().item<bool>())
            {
                score.g_mean = g_mean_exp.item().toDouble();
            }

            if(not torch::isnan(hmean_score).any().item<bool>()
               and not torch::isinf(hmean_score).any().item<bool>())
            {
                score.h_mean = hmean_score.item().toDouble();
            }
        }

        if(batch_payload.loss.numel())
        {
            const auto& loss = batch_payload.loss;

            const auto neg_log_likelihood = loss * torch::Scalar(int64_t(metadata.trg_len));
            score.negative_log_likelihood = neg_log_likelihood.item().toDouble();

            const auto loss_exp = loss.exp();
            if(not torch::isnan(loss_exp).any().item<bool>()
               and not torch::isinf(loss_exp).any().item<bool>())
            {
                score.loss = loss_exp.item().toDouble();
            }

            score.sentence_probability =
                metadata.current_actual_token_end_loc > 0
                    ? -1
                          * std::exp(-1 * loss.item().toDouble()
                                     * (metadata.current_actual_token_end_loc - 1))
                    : 0;
        }
        total_score.emplace_back(std::move(score));
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

std::vector<Vocinity::Context_Scorer::Score>
Vocinity::Context_Scorer::score_contexts(const std::vector<std::string>& sentences,
                                         const bool per_char_normalized)
{
    const std::lock_guard lock(_instance_mutex);
    struct Batch_Metadata
    {
        ulong actual_sequence_length = 0;
        torch::Tensor input_ids;
        torch::Tensor att_mask;
    };

    const torch::Tensor past = _inference_backend->get_initial_single_batch_past();
    std::vector<Batch_Metadata> batch_metadata;

    const auto pad_token_id = _tokenizer->get_pad_token_id();
    auto pad_and_push_tensor_to_stack =
        [&pad_token_id](torch::Tensor& stack, const torch::Tensor& tensor)
    {
        torch::Tensor better_tensor = tensor.unsqueeze(0);
        if(stack.size(-1) not_eq tensor.size(-1))
        {
            if(stack.size(-1) > tensor.size(-1))
            {
                better_tensor =
                    torch ::nn::functional::pad(better_tensor,
                                                torch::nn::functional::PadFuncOptions(
                                                    {0, stack.size(-1) - tensor.size(-1)})
                                                    .mode(torch::kConstant)
                                                    .value(pad_token_id));
            }
            else
            {
                stack = torch ::nn::functional::pad(stack,
                                                    torch::nn::functional::PadFuncOptions(
                                                        {0, tensor.size(-1) - stack.size(-1)})
                                                        .mode(torch::kConstant)
                                                        .value(pad_token_id));
            }
        }
        stack = torch::cat({stack, better_tensor}, 0);
    };

    torch::Tensor batched_input_ids, batched_att_masks, batched_labels, batched_pasts;
    for(uint sentence_order = 0; sentence_order < sentences.size(); ++sentence_order)
    {
        const auto& sentence                                  = sentences.at(sentence_order);
        const auto [input_ids, input_mask, actual_token_size] = encode(sentence);
        const unsigned long sequence_length                   = input_ids.size(-1);

        auto target_ids = input_ids.clone();
        target_ids.index_put_({Slice(None, -sequence_length)},
                              _inference_backend->get_label_ignore_id());

        if(batched_input_ids.numel())
        {
            pad_and_push_tensor_to_stack(batched_input_ids, input_ids);
            pad_and_push_tensor_to_stack(batched_att_masks, input_mask);
            pad_and_push_tensor_to_stack(batched_labels, target_ids);
#ifdef ONNX_AVAILABLE
            batched_pasts = torch::cat({batched_pasts, past}, 2);
#endif
        }
        else
        {
            batched_input_ids = input_ids.unsqueeze(0);
            batched_att_masks = input_mask.unsqueeze(0);
            batched_labels    = target_ids.unsqueeze(0);
            batched_pasts     = past;
        }

        batch_metadata.push_back({actual_token_size + 2, input_ids, input_mask});
    }

#ifdef CUDA_AVAILABLE
    if(_flush_torch_cuda_cache)
    {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif

    std::vector<Score> scores;
    const auto batch_payload = _inference_backend->score(
        batched_input_ids, batched_att_masks, batched_labels, batched_pasts);
    for(uint batch_order = 0; batch_order < batch_metadata.size(); ++batch_order)
    {
        Score score;
        score.utterance = sentences.at(batch_order);

        const auto& metadata = batch_metadata.at(batch_order);
        if(batch_payload.logits.numel())
        {
            const auto& logits    = batch_payload.logits[batch_order];
            const auto& input_ids = metadata.input_ids;

            const auto log_probs = torch::nn::functional::log_softmax(
                logits, torch::nn::functional::LogSoftmaxFuncOptions(-1));

            const auto& out_mask =
                input_ids.index({Slice(1, metadata.actual_sequence_length)});
            torch::Tensor target_log_probs =
                log_probs.gather(-1, out_mask.unsqueeze(-1)).squeeze(-1);

            const auto prod_score  = target_log_probs.sum();
            const auto mean_score  = metadata.actual_sequence_length > 0
                                         ? target_log_probs.logsumexp(0)
                                              - std::log(metadata.actual_sequence_length - 1)
                                         : torch::tensor(0);
            const auto gmean_score = target_log_probs.mean(0);
            const auto hmean_score = metadata.actual_sequence_length > 0
                                         ? target_log_probs.neg().logsumexp(0).neg()
                                               + std::log(metadata.actual_sequence_length - 1)
                                         : torch::tensor(0);

            score.production = prod_score.item().toDouble();

            const auto mean_exp = mean_score.exp();
            if(not torch::isnan(mean_exp).any().item<bool>()
               and not torch::isinf(mean_exp).any().item<bool>())
            {
                score.mean = mean_exp.item().toDouble();
            }

            const auto g_mean_exp = gmean_score.exp();
            if(not torch::isnan(g_mean_exp).any().item<bool>()
               and not torch::isinf(g_mean_exp).any().item<bool>())
            {
                score.g_mean = g_mean_exp.item().toDouble();
            }

            if(not torch::isnan(hmean_score).any().item<bool>()
               and not torch::isinf(hmean_score).any().item<bool>())
            {
                score.h_mean = hmean_score.item().toDouble();
            }
        }

        if(batch_payload.loss.numel())
        {
            const auto& loss = batch_payload.loss;

            const auto neg_log_likelihood =
                loss * torch::Scalar(int64_t(metadata.actual_sequence_length));
            score.negative_log_likelihood = neg_log_likelihood.item().toDouble();

            const auto loss_exp = loss.exp();
            if(not torch::isnan(loss_exp).any().item<bool>()
               and not torch::isinf(loss_exp).any().item<bool>())
            {
                score.loss = loss_exp.item().toDouble();
            }

            score.sentence_probability =
                metadata.actual_sequence_length > 0
                    ? -1
                          * std::exp(-1 * loss.item().toDouble()
                                     * (metadata.actual_sequence_length - 1))
                    : 0;
        }

        if(per_char_normalized)
        {
            score.negative_log_likelihood /= (metadata.actual_sequence_length + 1);
            score.production /= (metadata.actual_sequence_length + 1);
            score.mean /= (metadata.actual_sequence_length + 1);
            score.g_mean /= (metadata.actual_sequence_length + 1);
            score.h_mean /= (metadata.actual_sequence_length + 1);
            score.loss /= (metadata.actual_sequence_length + 1);
            score.sentence_probability /= (metadata.actual_sequence_length + 1);
        }

        scores.emplace_back(std::move(score));
    }

    return scores;
}

ushort
Vocinity::Context_Scorer::get_max_sequence_length() const
{
    return _inference_backend->get_max_sequence_length();
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
        std::transform(std::execution::unseq,
                       tokens.cbegin(),
                       tokens.cend(),
                       ids.begin() + 1,
                       [this](const auto& token) -> int64_t
                       { return _tokenizer->convert_token_to_id(token); }); // moves
#else
        __gnu_parallel::transform(tokens.begin(),
                                  tokens.end(),
                                  ids.begin() + 1,
                                  [this](const auto& token)
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
    {
        ids[0]              = _tokenizer->get_bos_token_id();
        ids[ids.size() - 1] = _tokenizer->get_eos_token_id();
    }
    const auto& tokens_padded = akil::memory::vector_1d_to_tensor_1d_no_copy<int64_t>(ids);
    const uint64_t lr_bos_eos_padded_token_size = ids.size();

    const uint64_t full_sequence_size =
        (lr_bos_eos_padded_token_size % _inference_backend->get_max_sequence_length())
            ? _inference_backend->get_max_sequence_length()
                  - (lr_bos_eos_padded_token_size
                     % _inference_backend->get_max_sequence_length())
                  + lr_bos_eos_padded_token_size
            : lr_bos_eos_padded_token_size;

    torch::Tensor full_sequence;

    {
        full_sequence = torch::full(
            full_sequence_size,
            _tokenizer->get_pad_token_id(),
            torch::TensorOptions().dtype(_inference_backend->get_input_int_range()));
    }

    full_sequence.index_put_({Slice(None, lr_bos_eos_padded_token_size)}, tokens_padded);


    auto input_mask =
        torch::zeros(full_sequence_size,
                     torch::TensorOptions().dtype(_inference_backend->get_input_int_range()));
    input_mask.index_put_({Slice(None, lr_bos_eos_padded_token_size)}, 1);

    return {full_sequence.to(_torch_device), input_mask.to(_torch_device), actual_token_size};
}
