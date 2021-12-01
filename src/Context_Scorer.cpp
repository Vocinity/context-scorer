#include "Context_Scorer.hpp"

#include "Tokenizer.cpp"

class Vocinity::Context_Scorer::Scorer_Backend
{
  public:
    virtual ~Scorer_Backend() = default;

  public:
    virtual std::pair<torch::Tensor, torch::Tensor> score(
        const at::Tensor& input_ids,
        const at::Tensor& att_mask,
        const torch::Tensor& labels = torch::Tensor()) = 0;

    virtual constexpr ushort get_max_sequence_length() = 0;

    virtual constexpr int64_t get_label_ignore_id() = 0;

    virtual constexpr int64_t get_stride() = 0;
};


#include <torch/csrc/api/include/torch/nn/functional/loss.h>

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
    std::pair<torch::Tensor, torch::Tensor> score(const at::Tensor& input_ids,
                                                  const at::Tensor& att_mask,
                                                  const torch::Tensor& labels) override
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
        return {logits, loss};
    }

    virtual constexpr ushort get_max_sequence_length() override
    {
        return 1024;
    }

    virtual constexpr int64_t get_label_ignore_id() override
    {
        return -100;
    }

    virtual constexpr int64_t get_stride() override
    {
        return 512;
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
        cudaStreamCreate(&_stream);
        cudaStreamCreate(&_cache_stream);
        cublasCreate(&_cublas_handler);
        cublasSetStream(_cublas_handler, _stream);

        // saved in custom proto file
        std::string res = _weights.initializing(scorer_model_path);
        if(!res.empty())
        {
            throw std::runtime_error(res);
        }

        _input             = std::vector<int>(max_batch_size * _weights._max_step, 0);
        _generation_buffer = std::vector<int>(max_batch_size * _weights._max_step, 0);
        _loss              = std::vector<float>(max_batch_size, 0.f);

        _encoder = std::make_shared<lightseq::cuda::GptEncoder<optype>>(
            max_batch_size,
            reinterpret_cast<int*>(thrust::raw_pointer_cast(_input.data())),
            reinterpret_cast<float*>(thrust::raw_pointer_cast(_loss.data())),
            reinterpret_cast<int*>(thrust::raw_pointer_cast(_generation_buffer.data())),
            _weights,
            _stream,
            _cache_stream,
            _cublas_handler);
        res = _encoder->check();
        if(!res.empty())
        {
            throw std::runtime_error(res);
        }

        long buf_bytesize = _encoder->compute_buffer_bytesize();
        _internal_buffer  = std::vector<int>(buf_bytesize / sizeof(int) + 1, 0);
        _encoder->init_buffer(
            reinterpret_cast<void*>(thrust::raw_pointer_cast(_internal_buffer.data())));
        cudaStreamSynchronize(_stream);
    }

    ~Scorer_LightSeq_Backend() override = default;

  public:
    std::pair<torch::Tensor, torch::Tensor> score(const at::Tensor& input_ids,
                                                  const at::Tensor& att_mask,
                                                  const torch::Tensor& labels) override
    {
        const Batch_Size batch_size       = input_ids.size(0);
        const Batch_Seq_Len batch_seq_len = input_ids.size(1);

#		ifdef QT_DEBUG
        auto start = std::chrono::high_resolution_clock::now();
#		endif

        cudaMemcpyAsync(reinterpret_cast<int*>(thrust::raw_pointer_cast(_input.data())),
                        input_ids.data_ptr(),
                        sizeof(int) * batch_size * batch_seq_len,
                        cudaMemcpyDeviceToDevice,
                        _stream);

        const auto logits_ptr = _encoder->run_one_infer(batch_size, batch_seq_len);

        c10::ScalarType precision;
#		ifdef CUDA_FP16_AVAILABLE
        if(optype == lightseq::cuda::OperationType::FP16)
        {
            precision = torch::kFloat16;
        }
        else if(optype == lightseq::cuda::OperationType::FP32)
#		endif
        {
            precision = torch::kFloat32;
        }

        const torch::Tensor logits =
            torch::from_blob(logits_ptr,
                             {batch_size, batch_seq_len},
                             torch::TensorOptions().device(torch::kCUDA).dtype(precision));

        const torch::Tensor losses = torch::from_blob(
            reinterpret_cast<int*>(thrust::raw_pointer_cast(_loss.data())),
            {batch_size},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

#		ifdef QT_DEBUG
        lightseq::cuda::print_time_duration(start, "one infer time", _stream);
        lightseq::cuda::print_vec(_loss.data(), "ppl", batch_size);
#		endif
        return {logits, losses};
    }

    virtual constexpr ushort get_max_sequence_length() override
    {
        return 512;
    }

    virtual constexpr int64_t get_label_ignore_id() override
    {
        return -100;
    }

    virtual constexpr int64_t get_stride() override
    {
        return 256;
    }

  private:
    cudaStream_t _stream;
    cudaStream_t _cache_stream;
    cublasHandle_t _cublas_handler;
    lightseq::cuda::GptWeight<optype> _weights;
    std::shared_ptr<lightseq::cuda::GptEncoder<optype>> _encoder;
    thrust::device_vector<int> _input;
    thrust::device_vector<int> _generation_buffer;
    thrust::device_vector<float> _loss;
    thrust::device_vector<int> _internal_buffer;
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
#		ifdef FASTER_TRANSFORMER_AVAILABLE
#		else
        _inference_backend = std::make_unique<Scorer_Torch_Backend>(scorer_model_path, device);
#		endif
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
            const auto& out_mask =
                input_ids.index({Slice(std::min(std::max((unsigned long) 1, begin_loc),
                                                current_actual_token_end_loc),
                                       current_actual_token_end_loc)});
            target_log_probs = log_probs.gather(-1, out_mask);
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
