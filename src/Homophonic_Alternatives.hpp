#ifndef HOMOPHONIC_ALTERNATIVES_HPP
#define HOMOPHONIC_ALTERNATIVES_HPP

#include <akil/aMisc.hpp>

//#define PROFILE_TIMING 1

namespace Vocinity
{
    class Homophonic_Alternative_Composer
    {
      private:
        class Homophonic_Alternative_Composer_Impl;

      public:
#ifdef ROBIN_HOOD_HASHING_AVAILABLE
#	define Unordered_Map robin_hood::unordered_flat_map
#else
#define Unordered_Map std::unordered_map
#endif
        using Word           = std::string;
        using Pronounciation = std::string;
        using Distance       = ushort;
        /**
     * + is addition, - is deletion, ~ is either nothing (when distance is zero) or substitution.
     */
        using Op                            = std::string;
        using Alternative_Word              = std::tuple<std::string, Distance, Op>;
        using Word_Alternatives             = std::vector<Alternative_Word>;
        using Alternative_Words_Of_Sentence = std::vector<Word_Alternatives>;

        enum class Matching_Method: short
    {
        Phoneme_Transcription=0 // Most Accurate and Slow
#ifdef LEVENSHTEIN_AVAILABLE
        ,Phoneme_Levenshtein=1 // Slowest and Accurate
#endif
#ifdef SOUNDEX_AVAILABLE
        ,Soundex=2 // Faster, Least Accurate
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
        ,Double_Metaphone=3 // Fastest, Less Acurate
#endif
    };

      public:
        struct Instructions
        {
            /** @brief max_best_num_alternatives should be set 0 for getting all. */
            ushort max_best_num_alternatives = 2;

            /** @brief max_distance is length of query word if -1 */
            short max_distance = 1;
            /**
         * @brief dismissed_word_indices will be used after splitting words by a single space.
         * Dismissed words wont be processed.
         */
            std::vector<ushort> dismissed_word_indices;
            /**
         * @brief dismissed_words wont be processed. Case insensitive. Can be used with
         * dismissed_word_indices together but using only one of them for this
         * purpose is enough.
         */
            std::vector<std::string> dismissed_words;
            /** @brief See Vocinity::Matching_Method::Homophonic_Alternative_Composer items for explanation. */
            Matching_Method method = Matching_Method::Phoneme_Transcription;
        };

      public:
        /** @brief <transcription,phonetic_encoding> is for phoneme matching and accepts cmudict encoding. */
        explicit Homophonic_Alternative_Composer(
            const std::vector<std::pair<std::string, std::string>>& phonetics_dictionary);

        ~Homophonic_Alternative_Composer();

      public:
        // https://cmusphinx.github.io/wiki/tutorialdict/
        // https://github.com/cmusphinx/g2p-seq2seq
        // https://github.com/AdolfVonKleist/Phonetisaurus
        static std::vector<std::pair<std::string, std::string>> load_phonetics_dictionary(
            const std::filesystem::path& dictionary = "./cmudict.0.7b.txt");

        static std::vector<std::vector<std::vector<std::pair<size_t, char>>>>
        load_precomputed_phoneme_similarity_map(
            const std::filesystem::path& map_path_to_be_imported,
                const bool binary = true);

        /** @brief Pre-index and just lookup whole cartesian relations which would be computed again and again otherwise. */
        void set_precomputed_phoneme_similarity_map(
            std::vector<std::vector<std::vector<std::pair<size_t, char>>>>&&
                map,const bool levenshtein=false);
        /**
     * @brief Takes too much time. dictionary is in-memory phonetics dictionary in cmudict form. */
        static std::vector<std::vector<std::vector<std::pair<size_t, char>>>>
        precompute_phoneme_similarity_map_from_phonetics_dictionary(
            const std::vector<std::pair<std::string, std::string>>& dictionary,
            const short max_distance               = -1,
            const ushort max_best_num_alternatives = 0,
            const bool levenshtein                 = false);

        static std::vector<std::vector<std::vector<std::pair<size_t, char>>>>
        precompute_phoneme_similarity_map_from_phonetics_dictionary(
            const std::filesystem::path& dictionary,
            const short max_distance               = -1,
            const ushort max_best_num_alternatives = 0,
            const bool levenshtein                 = false);

        static void save_precomputed_phoneme_similarity_map(
            const std::vector<std::vector<std::vector<std::pair<size_t, char>>>>& map,
            const std::filesystem::path& map_path_to_be_exported,
            const bool binary = true);

      public:
#ifdef SOUNDEX_AVAILABLE
        /**
     * @brief The dictionary is in <transcription,encoding> form.
     *  <transcription,soundex> is for Soundex
     *
     *  akil::string namespace contains Soundex encoder.
     */
        void set_in_memory_soundex_dictionary(
            const Unordered_Map<std::string, std::string>& dictionary);
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
        /**
     * @brief The dictionary is in <transcription,encoding> form.
     *  <transcription,<primary_code,alternative_code>> is for Metaphone.
     *
     *  akil::string namespace contains Double Metaphone encoder.
     */
        void set_in_memory_double_metaphone_dictionary(
            const Unordered_Map<std::string, std::pair<std::string, std::string>>& dictionary);
#endif
      public:
        Alternative_Words_Of_Sentence get_alternatives(const std::string& reference,
                                                       const Instructions& instructions,
                                                       const bool parallel = false);

      private:
        std::unique_ptr<Homophonic_Alternative_Composer_Impl> _impl;
    };
} // namespace Vocinity

#endif // HOMOPHONIC_ALTERNATIVES_HPP
