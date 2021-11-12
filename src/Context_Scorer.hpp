#ifndef CONTEXT_SCORER_HPP
#define CONTEXT_SCORER_HPP

#include <akil/aMisc.hpp>

namespace Vocinity
{
    class Homophonic_Alternative_Composer
    {
    private:
        class Homophonic_Alternative_Composer_Impl;

    public:
        using Word           = std::string;
        using Pronounciation = std::string;
        using Distance       = short;
        /**
         * + is addition, - is deletion, ~ is either nothing or substitution.
         */
        using Op                            = std::string;
        using Alternative_Word              = std::tuple<std::string, Distance, Op>;
        using Word_Alternatives             = std::vector<Alternative_Word>;
        using Alternative_Words_Of_Sentence = std::vector<Word_Alternatives>;

        enum class Matching_Method: short
        {
            Phoneme_Transcription=0
#ifdef LEVENSHTEIN_AVAILABLE
            ,Phoneme_Levenshtein=1
#endif
#ifdef SOUNDEX_AVAILABLE
            ,Soundex=2
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
            ,Double_Metaphone=3
#endif
        };

    public:
        struct Instructions
        {
            /**
             * @brief max_best_num_alternatives should be set 0 for getting all.
             */
            ushort max_best_num_alternatives = 0;

            short max_distance              = -1;
            /**
             * @brief dismissed_word_indices will be used after splitting words by a single space.
             */
            std::vector<uint64_t> dismissed_word_indices;
            /**
             * @brief dismissed_words wont be processed. Case insensitive.
             */
            std::vector<std::string> dismissed_words;
            Matching_Method method = Matching_Method::Phoneme_Transcription;
        };

    public:
        explicit Homophonic_Alternative_Composer(
                const std::filesystem::path& dictionary = "./cmudict.0.7a.txt");
        ~Homophonic_Alternative_Composer();

    public:
#ifdef SOUNDEX_AVAILABLE
        /**
         * @brief
         *  <transcription,phonetic_encoding> is for phoneme matching and accepts cmudict encoding.
         */
        void set_in_memory_soundex_dictionary(
                const std::unordered_map<std::string, std::string>& dictionary);
#endif
        /**
         * @brief The dictionary is in <transcription,encoding> form.
         *  <transcription,soundex> is for Soundex
         *
         *  akil::string namespace contains Soundex encoder.
         */
        void set_in_memory_phonemes_dictionary(
                const std::unordered_map<std::string, std::string>& dictionary);
#ifdef DOUBLE_METAPHONE_AVAILABLE
        /**
         * @brief The dictionary is in <transcription,encoding> form.
         *  <transcription,<primary_code,alternative_code>> is for Metaphone.
         *
         *  akil::string namespace contains Double Metaphone encoder.
         */
        void set_in_memory_double_metaphone_dictionary(
                const std::unordered_map<std::string, std::pair<std::string, std::string>>&
                dictionary);
#endif
    public:
        Alternative_Words_Of_Sentence get_alternatives(const std::string& reference,
                                                       const Instructions& instructions,
                                                       const bool parallel = false);

    private:
        std::unique_ptr<Homophonic_Alternative_Composer_Impl> _impl;
    };

    class Tokenizer;
    class Context_Scorer
    {
    private:
        class Scorer_Backend;

    public:
        enum class Inference_Backend : short { CPU = 0, CUDA = 1 };
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
                const Tokenizer_Configuration& encoding_conf = {}
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
        at::Tensor process_labels(const torch::Tensor& labels, const torch::Tensor& logits);
        Encoded_Sequence encode(const std::string& sentence, const bool parallel = false);

    private:
        std::unique_ptr<Scorer_Backend> _torch_runtime;
        c10::DeviceType _device = torch::kCPU;
        std::mutex _instance_mutex;
        std::unique_ptr<Tokenizer> _tokenizer;
        Model_Family _family = Model_Family::OpenAI;
    };
} // namespace Vocinity

#endif // CONTEXT_SCORER_HPP
