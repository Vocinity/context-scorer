#ifndef LIGHTSEQ_SCORER_BACKEND_HPP
#define LIGHTSEQ_SCORER_BACKEND_HPP

#ifdef CUDA_AVAILABLE
#	ifdef LIGHTSEQ_AVAILABLE
#include "Abstract-Scorer-Backend.hpp"

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
    Inference_Output score(const at::Tensor& input_ids,
                                                  const at::Tensor& att_mask,
                                                  const torch::Tensor& labels,
                                                  const torch::Tensor& past) override
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
        return {losses,logits,torch::Tensor()};
    }

    virtual ushort get_max_sequence_length() override
    {
        return 512;
    }

    virtual int64_t get_label_ignore_id() override
    {
        return -100;
    }

    virtual int64_t get_stride() override
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
#endif
