#ifndef CONTEXT-SCORER_H
#define CONTEXT-SCORER_H

#include <akil/aMisc.hpp>

namespace Vocinity
{
    class Tokenizer;
    class Context_Scorer
    {
      public:
        enum class Inference_Backend : short { CPU = 0, CUDA = 1 };
        enum Precision : short { FP32 = 0, FP16 = 1/*, INT8=2*/ };
        /**
         * OpenAI family is for GPT and all variants of GPT2 including distilgpt2.
         * All variants of GPT-Neo and 6J require you to set Neo as family.
         */
        enum class Model_Family : short { OpenAI = 0, Neo = 1 };

      public:
        using Input_Ids         = torch::Tensor;
        using Attention_Mask    = torch::Tensor;
        using Actual_Token_Size = uint64_t;
        using Encoded_Sequence  = std::tuple<Input_Ids, Attention_Mask, Actual_Token_Size>;

    public:
      class Scorer_Backend;

      public:
        struct Score
        {
            double negative_log_likelihood = 0;
            double production              = 0;
            double mean                    = 0;
            double g_mean                  = 0;
            double h_mean                  = 0;
            double loss                    = 0;
            double sentence_probability    = 0;
        };

        struct Tokenizer_Configuration
        {
            Tokenizer_Configuration(const std::filesystem::path& vocab_file_arg = {},
                                    const std::filesystem::path& merge_file_arg = {},
                                    const std::string& bos_token_str_arg = "<|endoftext|>",
                                    const std::string eos_token_str_arg  = "<|endoftext|>",
                                    const std::string pad_token_str_arg  = "<|endoftext|>",
                                    const std::string unk_token_str_arg  = "<|endoftext|>",
                                    const std::string mask_token_str_arg = "<|endoftext|>")
                : vocab_file(vocab_file_arg)
                , merge_file(merge_file_arg)
                , bos_token_str(bos_token_str_arg)
                , eos_token_str(eos_token_str_arg)
                , pad_token_str(pad_token_str_arg)
                , unk_token_str(unk_token_str_arg)
                , mask_token_str(mask_token_str_arg)
            {}
            std::filesystem::path vocab_file;
            std::filesystem::path merge_file;
            std::string bos_token_str;
            std::string eos_token_str;
            std::string pad_token_str;
            std::string unk_token_str;
            std::string mask_token_str;
        };

      public:
        static void optimize_parallelization_policy_for_use_of_multiple_instances();
        static void optimize_parallelization_policy_for_use_of_single_instance();

      public:
        explicit Context_Scorer(
            const std::filesystem::path& scorer_model_path,
            const Model_Family& family                   = Model_Family::OpenAI,
            const Tokenizer_Configuration& encoding_conf = {},
            const Precision precision=Precision::FP32
#ifdef CUDA_AVAILABLE
            ,
            const Inference_Backend device = Inference_Backend::CPU
#endif
        );

        Score score(const std::string& sentence, const bool per_char_normalized = false);

        virtual ~Context_Scorer();
        Context_Scorer(const Context_Scorer& other) = delete;
        Context_Scorer& operator=(const Context_Scorer&) = delete;

      private:
        Encoded_Sequence encode(const std::string& sentence, const bool parallel = false);

      private:
        std::unique_ptr<Scorer_Backend> _inference_backend;
        const Precision _precision;
        c10::DeviceType _device = torch::kCPU;
        std::mutex _instance_mutex;
        std::unique_ptr<Tokenizer> _tokenizer;
        Model_Family _family = Model_Family::OpenAI;
    };
} // namespace Vocinity

#endif // CONTEXT-SCORER_H
