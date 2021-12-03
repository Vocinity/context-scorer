#ifndef FASTER_TRANSOFRMER_SCORER_BACKEND_HPP
#define FASTER_TRANSOFRMER_SCORER_BACKEND_HPP

#ifdef CUDA_AVAILABLE
#	ifdef FASTER_TRANSFORMER_AVAILABLE
#		include "../3rdparty/FasterTransformer/src/fastertransformer/models/gpt/Gpt.h"
#		include "../3rdparty/FasterTransformer/src/fastertransformer/utils/cuda_utils.h"
#		include "../3rdparty/FasterTransformer/src/fastertransformer/utils/memory_utils.h"
#		include "Abstract-Scorer-Backend.hpp"
#		ifdef CUDA_FP16_AVAILABLE
#			include <cuda_fp16.h>
#		endif
#		ifdef QT_DEBUG
#			include <time.h>
#		endif

struct Faster_Transformer_Configuration
{
    size_t max_batch_size =
        8; //reader.GetInteger("ft_instance_hyperparameter", "max_batch_size");
    //const size_t max_seq_len =
    //    128; /*reader.GetInteger("ft_instance_hyperparameter", "max_seq_len");*/
    const size_t beam_width =
        1;                   //reader.GetInteger("ft_instance_hyperparameter", "beam_width");
    const int top_k   = 0;   //reader.GetInteger("ft_instance_hyperparameter", "top_k");
    const float top_p = 0.5; //reader.GetFloat("ft_instance_hyperparameter", "top_p");
    const float temperature =
        1.0; //reader.GetFloat("ft_instance_hyperparameter", "temperature");
    const float repetition_penalty =
        2.0; //reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty");
    const bool sparse =
        0; //static_cast<bool>(reader.GetInteger("ft_instance_hyperparameter", "sparse"));
    size_t request_batch_size = 8; //reader.GetInteger("request", "request_batch_size");
    // The length of tokens we hope this model to generate
    const int request_output_len = 32; //reader.GetInteger("request", "request_output_len");
};

struct Faster_Transformer_Model_Configuration
{
    enum class GPT2_MODELS : short{gpt_124M=0,gpt_175B=1,megatron_345M=2,megatron_6p7B=4};

    Faster_Transformer_Model_Configuration(
        const GPT2_MODELS model_type = GPT2_MODELS::gpt_124M)
    {
        if(model_type == GPT2_MODELS::gpt_124M)
        {
            head_num       = 12;
            size_per_head  = 64;
            vocab_size     = 50257;
            decoder_layers = 12;
        }
        else if(model_type == GPT2_MODELS::gpt_175B)
        {
            head_num       = 96;
            size_per_head  = 128;
            vocab_size     = 51200;
            decoder_layers = 96;
        }
        else if(model_type == GPT2_MODELS::megatron_345M)
        {
            head_num       = 16;
            size_per_head  = 64;
            vocab_size     = 50304;
            decoder_layers = 24;
        }
        else if(model_type == GPT2_MODELS::megatron_6p7B)
        {
            head_num       = 32;
            size_per_head  = 128;
            vocab_size     = 51200;
            decoder_layers = 32;
        }

        hidden_units = head_num * size_per_head;
        inter_size   = 4 * hidden_units;
    }
    size_t head_num;       // = reader.GetInteger(model_name, "head_num");
    size_t size_per_head;  // = reader.GetInteger(model_name, "size_per_head");
    size_t vocab_size;     // = reader.GetInteger(model_name, "vocab_size");
    size_t decoder_layers; // = reader.GetInteger(model_name, "decoder_layers");
    size_t hidden_units;
    size_t inter_size;
    static constexpr int inline beginning_of_text_token_id = 50256;
    static constexpr inline int end_of_text_token_id       = 50256;
};

template <typename precision = float>
class Scorer_FasterTransformer_Backend : public Vocinity::Context_Scorer::Scorer_Backend
{
    using Batch_Size    = int;
    using Batch_Seq_Len = int;

  public:
    Scorer_FasterTransformer_Backend(
        const std::filesystem::__cxx11::path& scorer_model_path,
        const Faster_Transformer_Model_Configuration::GPT2_MODELS gpt_model)
        : _model_configuration(Faster_Transformer_Model_Configuration(gpt_model))
        , allocator(fastertransformer::Allocator<fastertransformer::AllocatorType::CUDA>(
              fastertransformer::getDevice()))
#		ifdef SPARSITY_ENABLED
        , cublas_wrapper(fastertransformer::cublasMMWrapper(cublas_handle,
                                                            cublaslt_handle,
                                                            cusparselt_handle,
                                                            stream,
                                                            cublas_algo_map,
                                                            cublas_wrapper_mutex,
                                                            &allocator));
#		else
        , cublas_wrapper(fastertransformer::cublasMMWrapper(cublas_handle,
                                                            cublaslt_handle,
                                                            stream,
                                                            cublas_algo_map,
                                                            &instanceMutex,
                                                            &allocator))
#		endif
    {
        if(scorer_model_path.empty())
        {
            throw std::runtime_error("scorer_model_path can not be empty");
        }

        fastertransformer::check_cuda_error(cudaGetDeviceProperties(&prop, 0));
        printf("Device %s\n", prop.name);

        cudaStreamCreate(&stream);
        cublasCreate(&cublas_handle);
        cublasLtCreate(&cublaslt_handle);
        cublasSetStream(cublas_handle, stream);
#		ifdef SPARSITY_ENABLED
        CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
        cublas_algo_map = new cublasAlgoMap(GEMM_CONFIG, SPGEMM_CONFIG);
#		else
        cublas_algo_map = new fastertransformer::cublasAlgoMap(GEMM_CONFIG);
#		endif

#		ifdef CUDA_FP16_AVAILABLE
        if(std::is_same_v<precision, half>)
        {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
        else
#		endif
        {
            cublas_wrapper.setFP32GemmConfig();
        }
        gpt_weights =
            fastertransformer::GptWeight<precision>(_model_configuration.hidden_units,
                                                    _model_configuration.inter_size,
                                                    _model_configuration.vocab_size,
                                                    _model_configuration.decoder_layers,
                                                    get_max_input_sequence_length());

        gpt_weights.loadModel(scorer_model_path);

#		ifdef SPARSITY_ENABLED
        if(sparse)
        {
            printf("[INFO] Compress weights for sparse inference\n");
            gpt_weights.compress_weights(cublas_wrapper);
        }
#		endif

        max_output_seq_len = /////////////////////???//////////////////////////
            get_max_input_sequence_length() + _configuration.request_output_len;
        if(max_output_seq_len > (int) get_max_input_sequence_length())
        {
            throw std::runtime_error(
                "[ERROR] total_output_len " + std::to_string(max_output_seq_len)
                + " should be <= max_seq_len "
                + std::to_string(get_max_input_sequence_length()) + ". \n");
        }

        gpt = Gpt<precision>(0, // max_batch_size, FT will adjust the buffer automatically.
                             0, // max_seq_len, FT will adjust the buffer automatically.
                             0, // max_input_len, FT will adjust the buffer automatically.
                             _configuration.beam_width,
                             _model_configuration.head_num,
                             _model_configuration.size_per_head,
                             _model_configuration.inter_size,
                             _model_configuration.decoder_layers,
                             _model_configuration.vocab_size,
                             _model_configuration.beginning_of_text_token_id,
                             _model_configuration.end_of_text_token_id,
                             0.0f,
                             _configuration.top_k,
                             _configuration.top_p,
                             0,
                             _configuration.temperature,
                             1.0f, // len_penalty,
                             _configuration.repetition_penalty,
                             stream,
                             &cublas_wrapper,
                             &allocator,
                             false,
                             &prop,
                             _configuration.sparse);
        setup_buffers();
    }

    virtual ~Scorer_FasterTransformer_Backend() override
    {
        fastertransformer::deviceFree(d_output_ids);
        fastertransformer::deviceFree(d_parent_ids);
        fastertransformer::deviceFree(d_sequence_lengths);
        fastertransformer::deviceFree(d_input_ids);
        fastertransformer::deviceFree(d_input_lengths);
        fastertransformer::deviceFree(d_output_cum_log_probs);
#		ifdef SPARSITY_ENABLED
        cusparseLtDestroy(&cusparselt_handle);
#		endif
        delete cublas_algo_map;
    }

  public:
    std::pair<torch::Tensor, torch::Tensor> score(const at::Tensor& input_ids,
                                                  const at::Tensor& att_mask,
                                                  const torch::Tensor& labels) override
    {
        const Batch_Size batch_size       = input_ids.size(0);
        const Batch_Seq_Len batch_seq_len = input_ids.size(1);

        if(batch_size not_eq _configuration.request_batch_size)
        {
            _configuration.request_batch_size = batch_size;
            setup_buffers();
        }

        fastertransformer::check_cuda_error(
            cudaMemcpyAsync(d_input_ids,
                            input_ids.data_ptr(),
                            _configuration.request_batch_size * _configuration.beam_width
                                * get_max_input_sequence_length(),
                            cudaMemcpyDeviceToDevice,
                            _stream));

//        fastertransformer::check_cuda_error(
//            cudaMemcpyAsync(d_input_lengths,
//                            v_start_lengths.data_ptr(),
//                            _configuration.request_batch_size * _configuration.beam_width,
//                            cudaMemcpyDeviceToDevice,
//                            _stream));

#		ifdef QT_DEBUG
        fastertransformer::print_mem_usage();
        struct timeval start, end;
        gettimeofday(&start, NULL);
#		endif
        cudaDeviceSynchronize();

        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);

        cudaDeviceSynchronize();
#		ifdef QT_DEBUG
        gettimeofday(&end, NULL);
#		endif
#		ifdef QT_DEBUG
        printf("[INFO] request_batch_size %ld beam_width %ld head_num %ld size_per_head %ld "
               "total_output_len %d"
               " decoder_layers %ld vocab_size %ld FT-CPP-decoding-beamsearch-time %.2f ms\n",
               _configuration.request_batch_size,
               _configuration.beam_width,
               _model_configuration.head_num,
               _model_configuration.size_per_head,
               total_output_len,
               _model_configuration.decoder_layers,
               _model_configuration.vocab_size,
               ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001));
#		endif

        const size_t outCount =
            max_output_seq_len * _configuration.request_batch_size * _configuration.beam_width;

        c10::ScalarType torch_precision;
#		ifndef CUDA_FP16_AVAILABLE
        if(std::is_same_v<precision, half>)
        {
            torch_precision = torch::kFloat16;
        }
        else  if(std::is_same_v<precision, float>)
#		endif
        {
            torch_precision = torch::kFloat32;
        }

        const torch::Tensor generated =
            torch::from_blob(d_output_ids,
                             {max_output_seq_len , _configuration.request_batch_size , _configuration.beam_width},
                             torch::TensorOptions().device(torch::kCUDA).dtype(torch_precision));

        return {generated,generated};
    }

    virtual constexpr ushort get_max_sequence_length() override
    {
        return get_max_input_sequence_length();
    }

    virtual constexpr int64_t get_label_ignore_id() override
    {
        return -100;
    }

    virtual constexpr int64_t get_stride() override
    {
        return get_max_sequence_length() / 2;
    }

  private:
    static constexpr ushort get_max_input_sequence_length()
    {
        return 128;
    }

    void setup_buffers()
    {
        fastertransformer::deviceFree(d_output_ids);
        fastertransformer::deviceMalloc(&d_output_ids,
                                        _configuration.request_batch_size
                                            * _configuration.beam_width * max_output_seq_len,
                                        false);

        fastertransformer::deviceFree(d_parent_ids);
        fastertransformer::deviceMalloc(&d_parent_ids,
                                        _configuration.request_batch_size
                                            * _configuration.beam_width * max_output_seq_len,
                                        false);

        fastertransformer::deviceFree(d_sequence_lengths);
        fastertransformer::deviceMalloc(&d_sequence_lengths,
                                        _configuration.request_batch_size
                                            * _configuration.beam_width,
                                        false);

        fastertransformer::deviceFree(d_output_cum_log_probs);
        fastertransformer::deviceMalloc(&d_output_cum_log_probs,
                                        _configuration.request_batch_size
                                            * _configuration.beam_width
                                            * _configuration.request_output_len,
                                        false);

        input_tensors = std::vector<fastertransformer::Tensor>{
            fastertransformer::Tensor{
                fastertransformer::MEMORY_GPU,
                fastertransformer::TYPE_INT32,
                std::vector<size_t>{_configuration.request_batch_size
                                        * _configuration.beam_width,
                                    (size_t) get_max_input_sequence_length()},
                d_input_ids},
            fastertransformer::Tensor{fastertransformer::MEMORY_GPU,
                                      fastertransformer::TYPE_INT32,
                                      std::vector<size_t>{_configuration.request_batch_size
                                                          * _configuration.beam_width},
                                      d_input_lengths},
            fastertransformer::Tensor{fastertransformer::MEMORY_CPU,
                                      fastertransformer::TYPE_INT32,
                                      std::vector<size_t>{1},
                                      &max_output_seq_len}};

        output_tensors = std::vector<fastertransformer::Tensor>{
            fastertransformer::Tensor{fastertransformer::MEMORY_GPU,
                                      fastertransformer::TYPE_INT32,
                                      std::vector<size_t>{_configuration.request_batch_size,
                                                          _configuration.beam_width,
                                                          (size_t) max_output_seq_len},
                                      d_output_ids},
            fastertransformer::Tensor{fastertransformer::MEMORY_GPU,
                                      fastertransformer::TYPE_INT32,
                                      std::vector<size_t>{(size_t) max_output_seq_len,
                                                          _configuration.request_batch_size,
                                                          _configuration.beam_width},
                                      d_parent_ids},
            fastertransformer::Tensor{fastertransformer::MEMORY_GPU,
                                      fastertransformer::TYPE_INT32,
                                      std::vector<size_t>{_configuration.request_batch_size,
                                                          _configuration.beam_width},
                                      d_sequence_lengths},
            fastertransformer::Tensor{
                fastertransformer::MEMORY_GPU,
                fastertransformer::TYPE_FP32,
                std::vector<size_t>{(size_t) _configuration.request_output_len,
                                    _configuration.request_batch_size,
                                    _configuration.beam_width},
                d_output_cum_log_probs}};

        fastertransformer::deviceFree(d_input_ids);
        fastertransformer::deviceMalloc(&d_input_ids,
                                        _configuration.request_batch_size
                                            * _configuration.beam_width
                                            * get_max_input_sequence_length(),
                                        false);

        fastertransformer::deviceFree(d_input_lengths);
        fastertransformer::deviceMalloc(&d_input_lengths,
                                        _configuration.request_batch_size
                                            * _configuration.beam_width,
                                        false);

#		ifdef QT_DEBUG
        fastertransformer::print_mem_usage();
#		endif
    }

  private:
    cudaStream_t _stream;
    int max_output_seq_len = 0;
    std::vector<fastertransformer::Tensor> output_tensors;
    std::vector<fastertransformer::Tensor> input_tensors;
    int* d_output_ids           = nullptr;
    int* d_parent_ids           = nullptr;
    int* d_sequence_lengths     = nullptr;
    int* d_output_cum_log_probs = nullptr;
    fastertransformer::Gpt<precision> gpt;
    fastertransformer::GptWeight<precision> gpt_weights;
    fastertransformer::Allocator<fastertransformer::AllocatorType::CUDA> allocator;
#		ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    fastertransformer::cublasAlgoMap* cublas_algo_map = nullptr;
#		else
    fastertransformer::cublasAlgoMap* cublas_algo_map = nullptr;
#		endif
    fastertransformer::cublasMMWrapper cublas_wrapper;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    int* d_input_ids     = nullptr;
    int* d_input_lengths = nullptr;
    const Faster_Transformer_Model_Configuration _model_configuration;
    Faster_Transformer_Configuration _configuration;
    std::mutex instanceMutex;
    struct cudaDeviceProp prop;
};
#	endif
#endif


#endif
