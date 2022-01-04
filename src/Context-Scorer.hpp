#ifndef CONTEXT_SCORER_H
#define CONTEXT_SCORER_H

#include <akil/aMisc.hpp>

namespace Vocinity
{
    class Tokenizer;
    class Context_Scorer
    {
      public:
        enum class Inference_Hardware : short { CPU = 0, CUDA = 1 };
        enum Precision : short { FP32 = 0, FP16 = 1 /*, INT8=2*/ };
        enum class GPT_TYPE:short
        {
            DistilGPT2=0, // implemented
            GPT2_Small=1,
            GPT2_Medium=2,
            GPT2_Large=4,
            GPT2_XLarge=8,
            GPT_Neo=16,
            GPT_J=32,
            GPT_NEOX=64,
        };

      public:
        using Input_Ids         = torch::Tensor;
        using Attention_Mask    = torch::Tensor;
        using Actual_Token_Size = uint64_t;
        struct Encoded_Sequence
        {
            Input_Ids input_ids;
            Attention_Mask att_mask;
            Actual_Token_Size actual_token_size;
        };

      public:
        class Abstract_Scorer_Backend;

      public:
        struct Score
        {
            Score()
            {
                constexpr double min_double = std::numeric_limits<double>::min();
                negative_log_likelihood     = min_double;
                production                  = min_double;
                mean                        = min_double;
                g_mean                      = min_double;
                h_mean                      = min_double;
                loss                        = min_double;
                sentence_probability        = min_double;
            }
            double production = 0;
            double mean       = 0;
            double g_mean     = 0;
            double h_mean     = 0;
            /// global for batch, relatively unreliable metric when used for individual elements of a batch
            double negative_log_likelihood = 0;
            /// global for batch, relatively unreliable metric when used for individual elements of a batch
            double loss = 0;
            /// global for batch, relatively unreliable metric when used for individual elements of a batch
            double sentence_probability = 0;
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
        explicit Context_Scorer(const std::filesystem::path& scorer_model_path,
                                const GPT_TYPE type = GPT_TYPE::DistilGPT2,
                                const Tokenizer_Configuration& encoding_conf = {},
                                const Precision precision                    = Precision::FP32
#ifdef CUDA_AVAILABLE
                                ,
                                const Inference_Hardware device = Inference_Hardware::CPU
#endif
        );

        virtual ~Context_Scorer();
        Context_Scorer(const Context_Scorer& other) = delete;
        Context_Scorer& operator=(const Context_Scorer&) = delete;

      public:
        /**
         * @brief consider_intra_batching=true allows intra batching of one long single context by
         * dispatching parts of it once stacked as a batch.
         *
         * this is a good idea for decent gpu and long text but here the point:
         * -Especially in TensorRT, if your next time you run this function for
         * different multiplier of get_max_sequence_length characters (I hardcoded 64,
         * optimal max is 1024) then graph optimizer needs to profile that new different
         * shape of dynamic axis and should update engine cache.
         * This means you will lose seconds at the beginning of next run.
         * But if you split your sequential runs as equal length of 64 chars of blocks
         * or you are not planning to run this function again then no problem.
         * Note that, scores are only reproducible for same batch set.
         *
         *
         * consider_intra_batching=false has no such constrain and slower
         * if text is too long and your gpu is not saturated. It just runs
         * get_max_sequence_length chars long padded blocks of context one by one sequentially
         * without batching. So input is always in the [1,get_max_sequence_length] shape.
         * Summing up results of two blocks and inferencing two blocks at once is same for us
         * in terms of accuracy in our way of perplexity computation.
         *
         */
        Score score_context(const std::string& context,
                            const bool per_char_normalized     = true,
                            const bool consider_intra_batching = false);
        /**
         * @brief is batching perplexity computation of multiple separate contexts.
         * Note that, scores are only reproducible for same batch set.
         *
         * Scores will be similar for same item between single and batch runs but not
         * same.
         *
         * Result vector is in same order with the contexts vector/
         */
        std::vector<Score> score_contexts(const std::vector<std::string>& contexts,
                                          const bool per_char_normalized = true);

      public:
        ushort get_max_sequence_length() const;

      public:
#ifdef CUDA_AVAILABLE
        inline void flush_cuda_tensor_cache_before_inference(const bool flush = true)
        {
            _flush_torch_cuda_cache = flush;
        }
#endif

      private:
        Encoded_Sequence encode(const std::string& sentence, const bool parallel = true);
        Score score_short_context(const Encoded_Sequence& encoding,
                                  const bool per_char_normalized = false);
        Score score_long_context(const Encoded_Sequence& encoding,
                                 const bool per_char_normalized = false);

      private:
        bool _flush_torch_cuda_cache = false;
        const GPT_TYPE _type;
        std::unique_ptr<Abstract_Scorer_Backend> _inference_backend;
        const Precision _precision;
        c10::DeviceType _device = torch::kCPU;
        std::unique_ptr<Tokenizer> _tokenizer;
        std::mutex _instance_mutex;
    };
} // namespace Vocinity

#endif // CONTEXT-SCORER_H
