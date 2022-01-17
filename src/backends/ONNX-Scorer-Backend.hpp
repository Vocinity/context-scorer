#ifndef ONNX_SCORER_BACKEND_HPP
#define ONNX_SCORER_BACKEND_HPP


#ifdef ONNX_AVAILABLE
#	include "Abstract-Scorer-Backend.hpp"
#	include <onnx/onnxruntime_cxx_api.h>
#ifdef CENTOS_AVAILABLE
#	ifdef CUDA_AVAILABLE
#		ifdef TENSOR_RT_AVAILABLE
#			include <onnx/tensorrt_provider_factory.h>
#		endif
#	endif
#endif
#	include <torch/csrc/api/include/torch/nn/functional/loss.h>
// python -m onnxruntime.transformers.convert_to_onnx -m distilgpt2 --model_class GPT2LMHeadModel --output distilgpt2.onnx -p fp32 --use_gpu --verbose #--optimize_onnx // actually you dont need past hidden states for perplexity but does not matter
// python -m onnxruntime.transformers.optimizer --model_type gpt2 --input distilgpt2.onnx --output distilgpt2_optimized.onnx --num_heads 12 --hidden_size 768 --opt_level 1 --only_onnxruntime --verbose #--input_int32 --float16
//-------just know how to, dont use below-------
// python -m onnxruntime.tools.symbolic_shape_infer --input ./distilgpt2_optimized.onnx --output ./distilgpt2_optimized-shaped.onnx --verbose 3 --auto_merge
class Scorer_ONNX_Backend : public Vocinity::Context_Scorer::Abstract_Scorer_Backend
{
    using Batch_Size    = int;
    using Batch_Seq_Len = int;

  public:
    Scorer_ONNX_Backend(const std::filesystem::__cxx11::path& scorer_model_path,
                        const Vocinity::Context_Scorer::Precision precision =
                            Vocinity::Context_Scorer::Precision::FP32,
                        const unsigned long vocab_size = 50257,
                        const Vocinity::Context_Scorer::GPT_TYPE type =
                            Vocinity::Context_Scorer::GPT_TYPE::DistilGPT2
#	ifdef CUDA_AVAILABLE
                        ,
                        const Vocinity::Context_Scorer::Inference_Hardware device =
                            Vocinity::Context_Scorer::Inference_Hardware::CPU
#	endif
                        )
        : _vocab_size(vocab_size)
        , _type(type)
    {
        if(scorer_model_path.empty())
        {
            throw std::runtime_error("scorer_model_path can not be empty");
        }

#	ifdef CUDA_AVAILABLE
        _device = device;
#	endif
        _precision = precision;

        const auto providers = Ort::GetAvailableProviders();

        std::cout << "Available providers: ";
        std::copy(std::begin(providers),
                  std::end(providers),
                  std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << std::endl;

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        session_options.SetLogSeverityLevel(0);
        //   session_options.SetOptimizedModelFilePath(std::string(scorer_model_path.string()+"_runtime_optimized.onnx").c_str());
        session_options.AddConfigEntry("session.set_denormal_as_zero", "1");
        session_options.AddConfigEntry("optimization.enable_gelu_approximation", "1");
        //  session_options.EnableProfiling("profiling.prof");

#	ifdef CUDA_AVAILABLE
        if(device == Vocinity::Context_Scorer::Inference_Hardware::CUDA)
        {
#		ifdef TENSOR_RT_AVAILABLE
#ifdef CENTOS_AVAILABLE
           //Centos is buggy, you can not regiser TRTProvider options via cxx inline api. So you can not configure anything.
            OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);
#else
            OrtTensorRTProviderOptions trt_options;
            trt_options.device_id                    = 0;
            trt_options.has_user_compute_stream      = 0;
            trt_options.user_compute_stream          = nullptr;
            trt_options.trt_max_partition_iterations = 1000;
            trt_options.trt_min_subgraph_size        = 1;
            trt_options.trt_max_workspace_size       = 1 << 30;
            trt_options.trt_engine_cache_enable      = false; // true
            //trt_options.trt_engine_cache_path             = "";
            trt_options.trt_dump_subgraphs                = false; //
            trt_options.trt_engine_decryption_enable      = false;
            trt_options.trt_engine_decryption_lib_path    = "";
            trt_options.trt_force_sequential_engine_build = false;
#			ifdef CUDA_DLA_AVAILABLE
            trt_options.trt_dla_enable = true;
            trt_options.trt_dla_core   = 0;
#			else
            trt_options.trt_dla_enable                        = false;
            trt_options.trt_dla_core                          = 0;
#			endif
            // explicitly disable for now
//#			ifdef CUDA_INT8_AVAILABLE
//			trt_options.trt_int8_enable                       = true;
//			trt_options.trt_int8_calibration_table_name       = "";
//			trt_options.trt_int8_use_native_calibration_table = true;
//#			else
            trt_options.trt_int8_enable                       = false;
            trt_options.trt_int8_calibration_table_name       = "";
            trt_options.trt_int8_use_native_calibration_table = false;
//#			endif
#			ifdef CUDA_FP16_AVAILABLE
            if(precision == Vocinity::Context_Scorer::Precision::FP16)
            {
                trt_options.trt_fp16_enable = true;
            }
            else
            {
                trt_options.trt_fp16_enable = false;
            }
#			else
            trt_options.trt_fp16_enable                       = false;
#			endif
            std::cout<<"If it is crashing right after this line, one of possible reasons can be TensorRT could not bind your GPU."<<std::endl;
            session_options.AppendExecutionProvider_TensorRT(trt_options);
            std::cout<<"TensorRT execution provider succesfully registered and up, we are good."<<std::endl;
#endif
#		endif

            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy =
                1; // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
            cuda_options.gpu_mem_limit = 1L * 1024 * 1024 * 1024;
            std::cout << cuda_options.gpu_mem_limit
                      << " bytes of CuDNN workspace will be allocated." << std::endl;
            cuda_options.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            cuda_options.has_user_compute_stream   = 0;
            cuda_options.user_compute_stream       = nullptr;
            cuda_options.default_memory_arena_cfg  = nullptr;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
#	endif
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::cout << scorer_model_path.c_str() << std::endl;
        _session =
            std::make_unique<Ort::Session>(env, scorer_model_path.c_str(), session_options);
        _binding = std::make_unique<Ort::IoBinding>(*_session);

#	ifdef CUDA_AVAILABLE
        if(_device == Vocinity::Context_Scorer::Inference_Hardware::CUDA)
        {
            _cuda_memory_info = (Ort::MemoryInfo(
                "Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault));
        }
        else
#	endif
        {
            _cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                          OrtMemTypeDefault);
        }

        //#	ifdef QT_DEBUG
        std::cout << "Number of model inputs: " << _session->GetInputCount() << "\n";
        std::cout << "Number of model outputs: " << _session->GetOutputCount() << "\n";

        std::vector<int64_t> input_node_dims;
        const size_t num_input_nodes = _session->GetInputCount();
        std::vector<const char*> input_node_names(num_input_nodes);
        for(std::size_t i = 0; i < num_input_nodes; i++)
        {
            // print input node names
            char* input_name = _session->GetInputName(i, _default_allocator);
            std::cout << "Input" << i << " : "
                      << " name= " << input_name << std::endl;
            input_node_names[i] = input_name;
            // print input node types
            Ort::TypeInfo type_info        = _session->GetInputTypeInfo(i);
            auto tensor_info               = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::cout << "Input" << i << " : "
                      << " type= " << type << std::endl;

            // print input shapes/dims
            input_node_dims = tensor_info.GetShape();
            std::cout << "Input" << i << " : num_dims= " << input_node_dims.size()
                      << std::endl;
            for(int j = 0; j < input_node_dims.size(); j++)
            {
                if(input_node_dims[j] < 0)
                    input_node_dims[j] = 1;
                std::cout << "Input" << i << " : dim " << j << "= " << input_node_dims[j]
                          << std::endl;
            }
            _default_allocator.Free(input_name);
        }

        std::vector<int64_t> output_node_dims;
        size_t num_output_nodes = _session->GetOutputCount();
        std::vector<const char*> output_node_names(num_output_nodes);
        for(std::size_t i = 0; i < num_output_nodes; i++)
        {
            // print output node names
            char* output_name = _session->GetOutputName(i, _default_allocator);
            std::cout << "Output" << i << " : "
                      << " name= " << output_name << std::endl;
            output_node_names[i] = output_name;
            // print input node types
            Ort::TypeInfo type_info        = _session->GetOutputTypeInfo(i);
            auto tensor_info               = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::cout << "Output" << i << " : "
                      << " type= " << type << std::endl;

            // print input shapes/dims
            output_node_dims = tensor_info.GetShape();
            std::cout << "Output" << i << " : num_dims= " << output_node_dims.size()
                      << std::endl;
            for(int j = 0; j < output_node_dims.size(); j++)
            {
                if(output_node_dims[j] < 0)
                    output_node_dims[j] = 1;
                std::cout << "Output" << i << " : dim " << j << "= " << output_node_dims[j]
                          << std::endl;
            }
            _default_allocator.Free(output_name);
        }
        //#	endif
        std::cout << "Everything is okay in initialization of " << scorer_model_path
                  << std::endl;

        const auto& [hidden_size, num_attention_heads, num_layers, _] =
            Abstract_Scorer_Backend::get_configuration(_type);

        _past = torch::zeros(
            {num_layers, 2, 1, num_attention_heads, 0, hidden_size / num_attention_heads},
            torch::TensorOptions()
                .dtype(torch::kFloat32/*get_torch_precision()*/)
                .device(get_torch_device(_device)));
    }
    Scorer_ONNX_Backend(const Scorer_ONNX_Backend& other) = delete;
    Scorer_ONNX_Backend& operator=(const Scorer_ONNX_Backend& other) = delete;

    ~Scorer_ONNX_Backend() override
    {}

  public:
    Inference_Output score(const at::Tensor& input_ids,
                           const at::Tensor& att_mask,
                           const torch::Tensor& labels,
                           const torch::Tensor& past) override
    {
        const Batch_Size batch_size                = input_ids.size(0);
        const Batch_Seq_Len actual_sequence_length = input_ids.size(-1);

        const auto& [hidden_size, num_attention_heads, num_layers, _] =
            Abstract_Scorer_Backend::get_configuration(_type);

        const torch::Tensor position_ids =
            (att_mask.cumsum(-1) - 1).masked_fill(att_mask == 0, 1);

#	ifdef CUDA_AVAILABLE
        auto& memory_info_in_use =
            _device == Vocinity::Context_Scorer::Inference_Hardware::CUDA ? _cuda_memory_info
                                                                          : _cpu_memory_info;
#	else
        auto& memory_info_in_use = _cpu_memory_info;
#	endif

        _binding->ClearBoundInputs();

        create_out_buffers(batch_size, actual_sequence_length);

        const Ort::Value input_ids_bound =
            Ort::Value::CreateTensor(memory_info_in_use,
                                     reinterpret_cast<int64_t*>(input_ids.data_ptr()),
                                     batch_size * actual_sequence_length,
                                     input_ids.sizes().data(),
                                     input_ids.sizes().size());
        _binding->BindInput("input_ids", input_ids_bound);

        const Ort::Value att_mask_bound =
            Ort::Value::CreateTensor(memory_info_in_use,
                                     reinterpret_cast<float*>(att_mask.data_ptr()),
                                     batch_size * actual_sequence_length,
                                     att_mask.sizes().data(),
                                     att_mask.sizes().size());
        _binding->BindInput("attention_mask", att_mask_bound);

        const Ort::Value position_ids_bound =
            Ort::Value::CreateTensor(memory_info_in_use,
                                     reinterpret_cast<int64_t*>(position_ids.data_ptr()),
                                     batch_size * actual_sequence_length,
                                     position_ids.sizes().data(),
                                     position_ids.sizes().size());
        _binding->BindInput("position_ids", position_ids_bound);

        const uint past_elem_length =
            2 * batch_size * num_attention_heads * 1 * (hidden_size / num_attention_heads);
        for(int l = 0; l < num_layers; ++l)
        {
            const Ort::Value past_l_bound =
                Ort::Value::CreateTensor(memory_info_in_use,
                                         reinterpret_cast<float*>(past[l].data_ptr()),
                                         past_elem_length,
                                         past[l].sizes().data(),
                                         past[l].sizes().size());
            _binding->BindInput((std::string("past_") + std::to_string(l)).c_str(),
                                past_l_bound);
        }

        //  _binding->SynchronizeInputs();
        _session->Run(Ort::RunOptions(), *_binding);
        //  _binding->SynchronizeOutputs();

        const auto& shift_logits =
            logits.index({"...", Slice(None, -1), Slice()}).contiguous();
        const auto& shift_labels = labels.index({"...", Slice(1, None)}).contiguous();
        const auto loss          = torch::nn::functional::cross_entropy(
            shift_logits.view({-1, shift_logits.size(-1)}), shift_labels.view({-1}));
        return {loss, logits, present_states};
    }

    virtual ushort get_max_sequence_length() override
    {
        return std::max(max_input_sequence_length / 16, 64);
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
        if(_precision == Vocinity::Context_Scorer::Precision::FP16)
        {
            precision = torch::kFloat16;
        }
        else if(_precision == Vocinity::Context_Scorer::Precision::FP32)
        {
            precision = torch::kFloat32;
        }
        return precision;
    }

  private:
    void create_out_buffers(const Batch_Size batch_size       = 1,
                            const Batch_Seq_Len batch_seq_len = 0)
    {
        if(_batch_size == batch_size and batch_seq_len == _batch_seq_len)
        {
            return;
        }
        _batch_size    = batch_size;
        _batch_seq_len = batch_seq_len;

        _binding->ClearBoundOutputs();

#	ifdef CUDA_AVAILABLE
        auto& memory_info_in_use =
            _device == Vocinity::Context_Scorer::Inference_Hardware::CUDA ? _cuda_memory_info
                                                                          : _cpu_memory_info;
#	else
        auto& memory_info_in_use = _cpu_memory_info;
#	endif

        const c10::ScalarType precision = torch::kFloat32;//get_torch_precision();

        const auto& [hidden_size, num_attention_heads, num_layers, _] =
            Abstract_Scorer_Backend::get_configuration(_type);

        logits = torch::zeros(
            {_batch_size, _batch_seq_len, _vocab_size},
            torch::TensorOptions().dtype(precision).device(get_torch_device(_device)));
        present_states = torch::zeros(
            {num_layers,
             2,
             _batch_size,
             num_attention_heads,
             _batch_seq_len,
             hidden_size / num_attention_heads},
            torch::TensorOptions().dtype(precision).device(get_torch_device(_device)));
        const uint present_elem_length = 2 * _batch_size * num_attention_heads * _batch_seq_len
                                         * (hidden_size / num_attention_heads);

        for(int l = 0; l < num_layers; ++l)
        {
            const Ort::Value present_l_bound = Ort::Value::CreateTensor(
                memory_info_in_use,
                reinterpret_cast<float*>(present_states[l].data_ptr()),
                present_elem_length,
                present_states[l].sizes().data(),
                present_states[l].sizes().size());
            _binding->BindOutput((std::string("present_") + std::to_string(l)).c_str(),
                                 present_l_bound);
        }

        Ort::Value logits_bound =
            Ort::Value::CreateTensor(memory_info_in_use,
                                     reinterpret_cast<float*>(logits.data_ptr()),
                                     _batch_size * _batch_seq_len * _vocab_size,
                                     logits.sizes().data(),
                                     logits.sizes().size());

        _binding->BindOutput("logits", logits_bound);
    }


  private:
    const Vocinity::Context_Scorer::GPT_TYPE _type;
    std::unique_ptr<Ort::IoBinding> _binding;
#	ifdef CUDA_AVAILABLE
    Ort::MemoryInfo _cuda_memory_info{nullptr};
#	endif
    Ort::MemoryInfo _cpu_memory_info{nullptr};
    static inline Ort::Env env;
    std::unique_ptr<Ort::Session> _session;
    Ort::AllocatorWithDefaultOptions _default_allocator;
    static inline constexpr ushort max_input_sequence_length = 1024;
    unsigned long _vocab_size                                = 50257;
    torch::Tensor logits, present_states;
    Batch_Size _batch_size       = 0;
    Batch_Seq_Len _batch_seq_len = 0;
};
#endif
#endif
