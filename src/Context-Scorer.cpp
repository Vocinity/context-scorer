#include "Tokenizer.cpp"
#include "backends/Faster_Transformer-Scorer-Backend.hpp"
#include "backends/LightSeq-Scorer-Backend.hpp"
#include "backends/ONNX-Scorer-Backend.hpp"
#include "backends/Torch-Scorer-Backend.hpp"

Vocinity::Context_Scorer::Context_Scorer(const std::filesystem::path& scorer_model_path,
                                         const Model_Family& family,
                                         const Tokenizer_Configuration& encoding_conf,
                                         const Precision precision
#ifdef CUDA_AVAILABLE
                                         ,
                                         const Inference_Backend device
#endif
                                         )
    : _precision(precision)
#ifdef CUDA_AVAILABLE
    , _device(device == Inference_Backend::CUDA ? torch::kCUDA : torch::kCPU)
#endif
    , _tokenizer(std::make_unique<Tokenizer>(encoding_conf.vocab_file,
                                             encoding_conf.merge_file,
                                             encoding_conf.bos_token_str,
                                             encoding_conf.eos_token_str,
                                             encoding_conf.pad_token_str,
                                             encoding_conf.unk_token_str,
                                             encoding_conf.mask_token_str))
    , _family(family)
{
#ifdef CUDA_AVAILABLE
    if(device == Inference_Backend::CUDA)
    {
#	ifdef ONNX_AVAILABLE
        _inference_backend = std::make_unique<Scorer_ONNX_Backend>(
            scorer_model_path, precision, _tokenizer->get_vocab_size(), device);
#	else
#		ifdef LIGHTSEQ_AVAILABLE
#			ifdef CUDA_FP16_AVAILABLE
        if(precision == Precision::FP16)
        {
            _inference_backend =
                std::make_unique<Scorer_LightSeq_Backend<lightseq::cuda::OperationType::FP16>>(
                    scorer_model_path);
        }
        else
#			endif
        {
            _inference_backend =
                std::make_unique<Scorer_LightSeq_Backend<lightseq::cuda::OperationType::FP32>>(
                    scorer_model_path);
        }
#		else
#			ifdef FASTER_TRANSFORMER_AVAILABLE
#				ifdef CUDA_FP16_AVAILABLE
        _inference_backend =
            std::make_unique<Scorer_FasterTransformer_Backend<half>>(scorer_model_path);
#				else
        _inference_backend =
            std::make_unique<Scorer_FasterTransformer_Backend<float>>(scorer_model_path);
#				endif
#			else
        _inference_backend = std::make_unique<Scorer_Torch_Backend>(scorer_model_path, device);
#			endif
#		endif
#	endif
    }
    else
#endif
    {
#ifdef ONNX_AVAILABLE
        _inference_backend = std::make_unique<Scorer_ONNX_Backend>(
            scorer_model_path, precision, _tokenizer->get_vocab_size());
#else
        _inference_backend = std::make_unique<Scorer_Torch_Backend>(scorer_model_path);
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
Vocinity::Context_Scorer::score(const std::string& sentence, const bool per_char_normalized)
{
    auto [input_ids, input_mask, actual_token_size] = encode(sentence);

    const unsigned long sequence_length = input_ids.size(-1);

    if(_family == Model_Family::Neo)
    {
        input_ids  = input_ids.unsqueeze(0);
        input_mask = input_mask.unsqueeze(0);
    }

    std::vector<Score> total_score;
    for(size_t i = 0; i < sequence_length; i += _inference_backend->get_stride())
    {
        const auto begin_loc = i;
        const auto end_loc =
            std::min(i + _inference_backend->get_max_sequence_length(), sequence_length);
        const auto current_actual_token_end_loc =
            std::min(end_loc, std::max(actual_token_size + 2 - begin_loc, (unsigned long) 0));
        const auto trg_len            = end_loc - i;
        const auto& current_input_ids = input_ids.index({Slice(begin_loc, end_loc)});
        const auto& current_att_mask  = input_mask.index({Slice(begin_loc, end_loc)});
        auto target_ids               = current_input_ids.clone();
        target_ids.index_put_({Slice(None, -trg_len)},
                              _inference_backend->get_label_ignore_id());

        torch::Tensor logits, loss;
#ifdef CUDA_AVAILABLE
        if(_device == torch::kCUDA)
        {
#	ifdef LIGHTSEQ_AVAILABLE
            if(_family == Model_Family::Neo)
            {
                throw std::runtime_error("GPT Neo is not implemented in LightSeq!");
            }
            else
            {
                const auto payload = _inference_backend->score(current_input_ids.unsqueeze(0),
                                                               current_att_mask.unsqueeze(0));
                logits             = payload.first.flatten();
                loss               = payload.second.flatten();
            }
#	else
#		ifdef FASTER_TRANSFORMER_AVAILABLE
            if(_family == Model_Family::Neo)
            {}
            else
            {}
#		else
            const auto payload =
                _inference_backend->score(current_input_ids, current_att_mask, target_ids);
            logits = payload.first;
            loss   = payload.second;
#		endif
#	endif
        }
        else
#endif
        {
            const auto payload =
                _inference_backend->score(current_input_ids, current_att_mask, target_ids);
            logits = payload.first;
            loss   = payload.second;
        }

        if(not logits.numel())
        {
            continue;
        }

        const auto log_probs = torch::nn::functional::log_softmax(
            logits, torch::nn::functional::LogSoftmaxFuncOptions(-1));

        torch::Tensor target_log_probs;
#ifdef CUDA_AVAILABLE
#	ifdef LIGHTSEQ_AVAILABLE
        if(_device == torch::kCUDA)
        {
            const auto& out_mask = input_ids.index({Slice(
                std::min(std::max((unsigned long) 1, begin_loc), current_actual_token_end_loc),
                current_actual_token_end_loc)});
            target_log_probs     = log_probs.gather(-1, out_mask);
        }
        else
#	endif
#endif
        {
            if(_family == Model_Family::OpenAI)
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
        }

        Score score;

        const auto neg_log_likelihood = loss * torch::Scalar(int64_t(trg_len));
        score.negative_log_likelihood = neg_log_likelihood.item().toDouble();

        const auto prod_score = target_log_probs.sum();
        score.production      = prod_score.item().toDouble();

        const auto mean_score =
            current_actual_token_end_loc > 0
                ? target_log_probs.logsumexp(0) - std::log(current_actual_token_end_loc - 1)
                : torch::tensor(0);
        score.mean = mean_score.exp().item().toDouble();

        const auto gmean_score = target_log_probs.mean(0);
        score.g_mean           = gmean_score.exp().item().toDouble();

        const auto hmean_score = current_actual_token_end_loc > 0
                                     ? target_log_probs.neg().logsumexp(0).neg()
                                           + std::log(current_actual_token_end_loc - 1)
                                     : torch::tensor(0);
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

Vocinity::Context_Scorer::Encoded_Sequence
Vocinity::Context_Scorer::encode(const std::string& sentence, const bool parallel)
{
    const auto tokens = _tokenizer->tokenize(sentence);
    std::vector<int64_t> ids;
    ids.resize(tokens.size() + 2);
#ifdef CPP17_AVAILABLE
    if(parallel)
    {
        std::transform(std::execution::unseq,
                       tokens.begin(),
                       tokens.end(),
                       ids.begin() + 1,
                       [this](const auto& token) -> int64_t
                       { return _tokenizer->convert_token_to_id(token); }); // moves
    }
    else
#endif
    {
        std::transform(tokens.begin(),
                       tokens.end(),
                       ids.begin() + 1,
                       [this](const auto& token)
                       { return _tokenizer->convert_token_to_id(token); }); // moves
    }
    const Actual_Token_Size actual_token_size = ids.size() - 2;
#ifdef CUDA_AVAILABLE
#	ifdef LIGHTSEQ_AVAILABLE
    if(_device == torch::kCUDA)
    {
        ids[0]              = 0;
        ids[ids.size() - 1] = 0;
    }
    else
#	endif
#endif
    {
        ids[0]              = _tokenizer->get_bos_token_id();
        ids[ids.size() - 1] = _tokenizer->get_eos_token_id();
    }
    const auto& tokens_padded = akil::memory::vector_1d_to_tensor_1d_no_copy<int64_t>(ids);
    const uint64_t padded_token_size = ids.size();

    const uint64_t full_sequence_size =
        (padded_token_size % _inference_backend->get_max_sequence_length())
            ? _inference_backend->get_max_sequence_length()
                  - (padded_token_size % _inference_backend->get_max_sequence_length())
                  + padded_token_size
            : padded_token_size;
    torch::Tensor full_sequence;
#ifdef CUDA_AVAILABLE
#	ifdef LIGHTSEQ_AVAILABLE
    if(_device == torch::kCUDA)
    {
        full_sequence =
            torch::full(full_sequence_size, 0, torch::TensorOptions().dtype(torch::kInt64));
    }
    else
#	endif
#endif
    {
        full_sequence = torch::full(full_sequence_size,
                                    _tokenizer->get_pad_token_id(),
                                    torch::TensorOptions().dtype(torch::kInt64));
    }

    full_sequence.index_put_({Slice(None, padded_token_size)}, tokens_padded);


    auto input_mask = torch::zeros(full_sequence_size);
    input_mask.index_put_({Slice(None, padded_token_size)}, 1);

    return {full_sequence.to(_device), input_mask.to(_device), actual_token_size};
}
