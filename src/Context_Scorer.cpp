#include "Context_Scorer.hpp"

#include "Tokenizer.cpp"

class Vocinity::Context_Scorer::Scorer_Backend
{
  public:
    virtual ~Scorer_Backend() = default;

  public:
    virtual at::Tensor score(const at::Tensor& input_ids, const at::Tensor& att_mask) = 0;

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
};

class Scorer_Torch_Backend : public Vocinity::Context_Scorer::Scorer_Backend
{
  public:
    Scorer_Torch_Backend(
        const std::filesystem::__cxx11::path& scorer_model_path = std::string()
#ifdef CUDA_AVAILABLE
            ,
        const Vocinity::Context_Scorer::Inference_Backend device =
            Vocinity::Context_Scorer::Inference_Backend::CPU
#endif
    )
    {
        if(scorer_model_path.empty())
        {
            throw std::runtime_error("scorer_model_path can not be empty");
        }
        _scorer_model = torch::jit::load(scorer_model_path.string());
#ifdef CUDA_AVAILABLE
        if(device == Vocinity::Context_Scorer::Inference_Backend::CUDA)
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
    at::Tensor score(const at::Tensor& input_ids, const at::Tensor& att_mask) override
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

  private:
    torch::jit::script::Module _scorer_model;
    std::mutex instanceMutex;
    c10::InferenceMode guard{true};
};

#ifdef CUDA_AVAILABLE
#	ifdef LIGHTSEQ_AVAILABLE
#		ifdef QT_DEBUG
#			define DEBUG_RESULT
#		endif
#		include "../3rdparty/lightseq/lightseq/inference/model/gpt_encoder.h"
#		include "../3rdparty/lightseq/lightseq/inference/tools/util.h"

template <lightseq::cuda::OperationType optype>
class Scorer_LightSeq_Backend : public Vocinity::Context_Scorer::Scorer_Backend
{
    using Batch_Size    = int;
    using Batch_Seq_Len = int;

  public:
    Scorer_LightSeq_Backend(
        const std::filesystem::__cxx11::path& scorer_model_path = std::string(),
        const int max_batch_size                                = 128)
    {
        if(scorer_model_path.empty())
        {
            throw std::runtime_error("scorer_model_path can not be empty");
        }
        cudaSetDevice(0);
        cudaStreamCreate(&stream_);
        cudaStreamCreate(&cache_stream_);
        cublasCreate(&hd_);
        cublasSetStream(hd_, stream_);

        lightseq::cuda::GptWeight<optype> tw_;
        // saved in custom proto file
        std::string res = tw_.initializing(scorer_model_path);
        if(!res.empty())
        {
            throw std::runtime_error(res);
        }

        d_input_  = std::vector<int>(max_batch_size * tw_._max_step, 0);
        d_sample_ = std::vector<int>(max_batch_size * tw_._max_step, 0);
        d_ppl_    = std::vector<float>(max_batch_size, 0.f);

        encoder_ = std::make_shared<lightseq::cuda::GptEncoder<optype>>(
            max_batch_size,
            reinterpret_cast<int*>(thrust::raw_pointer_cast(d_input_.data())),
            reinterpret_cast<float*>(thrust::raw_pointer_cast(d_ppl_.data())),
            reinterpret_cast<int*>(thrust::raw_pointer_cast(d_sample_.data())),
            tw_,
            stream_,
            cache_stream_,
            hd_);
        res = encoder_->check();
        if(!res.empty())
        {
            throw std::runtime_error(res);
        }

        long buf_bytesize = encoder_->compute_buffer_bytesize();
        d_buf_            = std::vector<int>(buf_bytesize / sizeof(int) + 1, 0);
        encoder_->init_buffer(
            reinterpret_cast<void*>(thrust::raw_pointer_cast(d_buf_.data())));
        cudaStreamSynchronize(stream_);
    }

    ~Scorer_LightSeq_Backend() override = default;

  public:
    at::Tensor score(const at::Tensor& input_ids, const at::Tensor& att_mask) override
    {
        std::tuple<BetterCpp::span<int>, Batch_Size, Batch_Seq_Len> result;
        auto& [input_ids_vec, batch_size, batch_seq_len] = result;
        batch_size                                       = input_ids.size(0);
        batch_seq_len                                    = input_ids.size(1);
        input_ids_vec = akil::memory::tensor_1d_to_span_1d_no_copy<int>(input_ids);
        std::cout << input_ids.sizes() << std::endl;
#		ifdef QT_DEBUG
        auto start = std::chrono::high_resolution_clock::now();
#		endif
        // copy inputs from cpu memory to gpu memory
        cudaMemcpyAsync(reinterpret_cast<int*>(thrust::raw_pointer_cast(d_input_.data())),
                        input_ids_vec.data(),
                        sizeof(int) * batch_size * batch_seq_len,
                        cudaMemcpyHostToDevice,
                        stream_);
        encoder_->run_one_infer(batch_size, batch_seq_len);

#		ifdef QT_DEBUG
        lightseq::cuda::print_time_duration(start, "one infer time", stream_);
        lightseq::cuda::print_vec(d_ppl_.data(), "ppl", batch_size);
#		endif
        return torch::Tensor{};
    }

    static constexpr ushort get_max_sequence_length()
    {
        return 512;
    }

    static constexpr int64_t get_label_ignore_id()
    {
        return -100;
    }

    static constexpr int64_t get_stride()
    {
        return 256;
    }

  private:
    cudaStream_t stream_;
    cudaStream_t cache_stream_;
    cublasHandle_t hd_;
    std::shared_ptr<lightseq::cuda::GptEncoder<optype>> encoder_;
    thrust::device_vector<int> d_input_;
    thrust::device_vector<int> d_sample_;
    thrust::device_vector<float> d_ppl_;
    thrust::device_vector<int> d_buf_;
};
#	endif
#endif

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
#	ifdef LIGHTSEQ_AVAILABLE
#		ifdef CUDA_FP16_AVAILABLE
        if(precision == Precision::FP16)
        {
            _inference_backend =
                std::make_unique<Scorer_LightSeq_Backend<lightseq::cuda::OperationType::FP16>>(
                    scorer_model_path);
        }
        else
#		endif
        {
            _inference_backend =
                std::make_unique<Scorer_LightSeq_Backend<lightseq::cuda::OperationType::FP32>>(
                    scorer_model_path);
        }
#	else
        //
#	endif
    }
    else
#endif
    {
        _inference_backend = std::make_unique<Scorer_Torch_Backend>(scorer_model_path);
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

#include <torch/csrc/api/include/torch/nn/functional/loss.h>

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

        torch::Tensor logits;
#ifdef CUDA_AVAILABLE
        if(_device == torch::kCUDA)
        {
#	ifdef LIGHTSEQ_AVAILABLE
            if(_family == Model_Family::Neo)
            {
                logits = _inference_backend->score(current_input_ids, current_att_mask);
            }
            else
            {
                logits = _inference_backend->score(current_input_ids.unsqueeze(0),
                                                   current_att_mask.unsqueeze(0));
            }
#	else
#		ifdef FASTER_TRANSFORMER_AVAILABLE
            if(_family == Model_Family::Neo)
            {
                logits = _inference_backend->score(current_input_ids, current_att_mask);
            }
            else
            {
                logits = _inference_backend->score(current_input_ids.unsqueeze(0),
                                                   current_att_mask.unsqueeze(0));
            }
#		else
            current_input_ids.to(_device);
            current_att_mask.to(_device);
            logits = _inference_backend->score(current_input_ids, current_att_mask);
#		endif
#	endif
        }
        else
#endif
        {
            logits = _inference_backend->score(current_input_ids, current_att_mask);
        }

        if(not logits.numel())
        {
            continue;
        }

        const auto& loss      = process_labels(target_ids, logits);
        const auto& log_probs = torch::nn::functional::log_softmax(
            logits, torch::nn::functional::LogSoftmaxFuncOptions(-1));

        torch::Tensor target_log_probs;
        if(_family == Model_Family::OpenAI)
        {
            const auto& out_mask = input_ids.index({Slice(
                std::min(std::max((unsigned long) 1, begin_loc), current_actual_token_end_loc),
                current_actual_token_end_loc)});
            target_log_probs =
                log_probs.gather(-1, out_mask.unsqueeze(-1)).squeeze(-1).to(torch::kCPU);
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
            target_log_probs =
                log_probs.gather(-1, out_mask).squeeze(-1).squeeze(0).to(torch::kCPU);
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
        std::transform(std::execution::unseq,
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
        (padded_token_size % _inference_backend->get_max_sequence_length())
            ? _inference_backend->get_max_sequence_length()
                  - (padded_token_size % _inference_backend->get_max_sequence_length())
                  + padded_token_size
            : padded_token_size;
    auto full_sequence = torch::full(full_sequence_size,
                                     _tokenizer->get_pad_token_id(),
                                     torch::TensorOptions().dtype(torch::kInt64));
    full_sequence.index_put_({Slice(None, padded_token_size)}, tokens_padded);


    auto input_mask = torch::zeros(full_sequence_size);
    input_mask.index_put_({Slice(None, padded_token_size)}, 1);

    return {full_sequence, input_mask, actual_token_size};
}
