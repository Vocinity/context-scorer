#include "Context_Scorer.hpp"

class Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer_Impl
{
    using In_Memory_Dictionary = std::unordered_map<std::string, std::string>;

  public:
    explicit Homophonic_Alternative_Composer_Impl(const std::filesystem::path& dictionary)
    {
        if(not dictionary.empty())
        {
            std::ifstream dictionary_file(dictionary);

            constexpr std::string_view delim{"  "};
            for(std::string line, word, pronounciation; std::getline(dictionary_file, line);)
            {
                const auto parts = akil::string::split(line, delim.data());
                if(parts.size() == 2)
                {
                    word           = parts.at(0);
                    pronounciation = parts.at(1);
                }
                else
                {
                    std::cout << dictionary
                              << " is corrupted, 3 items expected in each line,\n entry of: |"
                              << line << "| is ignored";
                }
                const auto& phonemes    = get_phonemes(pronounciation);
                _phonemes_by_word[word] = {pronounciation, phonemes, phonemes.size()};
            }
        }
    }

    ~Homophonic_Alternative_Composer_Impl()
    {}

  public:
    void set_in_memory_phonemes_dictionary(const In_Memory_Dictionary& dictionary)
    {
        _phonemes_by_word.clear();
        for(const auto& [word, pronounciation] : dictionary)
        {
            const auto& phonemes    = get_phonemes(pronounciation);
            _phonemes_by_word[word] = {pronounciation, phonemes, phonemes.size()};
        }
    }
#ifdef SOUNDEX_AVAILABLE
    void set_in_memory_soundex_dictionary(const In_Memory_Dictionary& dictionary)
    {
        _soundex_encoding_by_word = dictionary;
    }
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
    void set_in_memory_double_metaphone_dictionary(
        const std::unordered_map<std::string, std::pair<std::string, std::string>>& dictionary)
    {
        _double_metaphone_encoding_by_word = dictionary;
    }
#endif
  public:
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives get_alternatives(
        const std::string& query_word,
        const Instructions& instructions,
        const bool parallel)
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;

        bool empty_dict = false;
        bool exists     = false;
        if(instructions.method == Matching_Method::Phoneme_Transcription)
        {
            empty_dict = _phonemes_by_word.empty();
            if(not empty_dict)
            {
                exists = _phonemes_by_word.contains(query_word);
            }
            else
            {
                result = get_alternatives_by_phoneme(query_word, instructions);
            }
        }
        else if(instructions.method == Matching_Method::Soundex)
        {
            empty_dict = _soundex_encoding_by_word.empty();
            if(not empty_dict)
            {
                exists = _soundex_encoding_by_word.contains(query_word);
            }
            else
            {
                result = get_alternatives_by_soundex(query_word);
            }
        }
        else if(instructions.method == Matching_Method::Double_Metaphone)
        {
            empty_dict = _double_metaphone_encoding_by_word.empty();
            if(not empty_dict)
            {
                exists = _double_metaphone_encoding_by_word.contains(query_word);
            }
            else
            {
                result = get_alternatives_by_metaphone(query_word);
            }
        }

        if(empty_dict)
        {
            std::cout << "Dictionary is empty. Either you had to pass dict path to ctor or "
                         "set in-memory dictionary";
            return {};
        }

        if(not exists)
        {
            return {};
        }

        if(instructions.max_best_num_alternatives)
        {
            if(result.size() > instructions.max_best_num_alternatives)
            {
                if(parallel)
                {
#ifdef CPP17_AVAILABLE
                    std::sort(std::execution::par_unseq,
                              result.begin(),
                              result.end(),
                              [](const Alternative_Word& one,
                                 const Alternative_Word& another) -> bool
                              {
                                  const auto& [_, first_distance, __]        = one;
                                  const auto& [dummy, second_distance, null] = another;
                                  return std::fabs(first_distance)
                                         < std::fabs(second_distance);
                              });
#else
                    __gnu_parallel::sort(
                        result.begin(),
                        result.end(),
                        [](const Alternative_Word& one,
                           const Alternative_Word& another) -> bool
                        {
                            const auto& [_, first_distance, __]        = one;
                            const auto& [dummy, second_distance, null] = another;
                            return std::fabs(first_distance) < std::fabs(second_distance);
                        });
#endif
                }
                else
                {
                    std::sort(result.begin(),
                              result.end(),
                              [](const Alternative_Word& one,
                                 const Alternative_Word& another) -> bool
                              {
                                  const auto& [_, first_distance, __]        = one;
                                  const auto& [dummy, second_distance, null] = another;
                                  return std::fabs(first_distance)
                                         < std::fabs(second_distance);
                              });
                }
                result.resize(instructions.max_best_num_alternatives);
            }
        }

        return result;
    }

  private:
#ifdef DOUBLE_METAPHONE_AVAILABLE
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives get_alternatives_by_metaphone(
        const std::string& query_word) const
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;

        const auto& [query_primary_code, query_alternative_code] =
            akil::string::get_double_metaphone_hash(query_word);
        for(const auto& [dictionary_word, dictionary_codes] :
            _double_metaphone_encoding_by_word)
        {
            const auto& [dictionary_primary_code, dictionary_alternate_code] =
                dictionary_codes;
            if(dictionary_primary_code == query_primary_code)
            {
                result.push_back({dictionary_word, 0, "~"});
                continue;
            }
            if(dictionary_alternate_code == query_alternative_code)
            {
                result.push_back({dictionary_word, 0, "~"});
            }
        }
        return result;
    }
#endif
#ifdef SOUNDEX_AVAILABLE
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives get_alternatives_by_soundex(
        const std::string& query_word) const
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;

        const auto& query_code = akil::string::get_soundex_hash(query_word);
        for(const auto& [dictionary_word, dictionary_code] : _soundex_encoding_by_word)
        {
            if(dictionary_code == query_code)
            {
                result.push_back({dictionary_word, 0, "~"});
            }
        }
        return result;
    }
#endif

    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives get_alternatives_by_phoneme(
        const std::string& query_word,
        const Instructions& instructions) const
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;

        const auto& [query_pronounciation, query_phonemes, query_phonemes_count] =
            _phonemes_by_word.at(query_word);
        for(const auto& [dictionary_word, dictionary_pronounciation_info] : _phonemes_by_word)
        {
            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = dictionary_pronounciation_info;
            if(dictionary_word_pronounciation == query_pronounciation
               and dictionary_word not_eq query_word)
            {
                result.push_back({query_word, 0, "~"});
                continue;
            }

            const ushort num_of_common_phonemes =
                get_num_of_common_phonemes(query_phonemes,
                                           query_phonemes_count,
                                           dictionary_word_phonemes,
                                           dictionary_word_phonemes_count);
            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');

            for(int distance = 1; distance < instructions.max_distance; ++distance)
            {
                //if the amount of similar Phonemes in w and the dictionary word is
                //equal to the the amount of phonemes in w, and the dictionary word has
                //one more phoneme total than w, add dictionary word to addPhoneme string
                if(num_of_common_phonemes == query_phonemes_count
                   and dictionary_word_phonemes_count == (query_phonemes_count + distance))
                {
                    if(!has_parenthesis)
                    { //ignores multiple pronunciations
                        result.push_back({dictionary_word, distance, "+"});
                    }
                }

                //if the amount of similar Phonemes in w and the dictionary word is
                //one less than the amount of phonemes in w, and the dictionary word has
                //one less total phonemes than w, add dictionary word to removePhoneme string
                else if(num_of_common_phonemes == query_phonemes_count - distance
                        and dictionary_word_phonemes_count
                                == (query_phonemes_count - distance))
                {
                    if(!has_parenthesis)
                    { //ignores multiple pronunciations
                        result.push_back({dictionary_word, -distance, "-"});
                    }
                }

                //if the amount of similar Phonemes in w and the dictionary word is one less
                //than the amount of phonemes in w, and the dictionary word has the same
                //amount of phonemes than w, add dictionary word to replacePhoneme string
                else if(num_of_common_phonemes == query_phonemes_count - distance
                        and dictionary_word_phonemes_count == query_phonemes_count)
                {
                    const auto ordered_common_phonemes =
                        get_num_of_common_phonemes(query_phonemes,
                                                   query_phonemes_count,
                                                   dictionary_word_phonemes,
                                                   dictionary_word_phonemes_count,
                                                   true);
                    if(ordered_common_phonemes == query_phonemes_count - distance)
                    {
                        if(!has_parenthesis)
                        { //ignores multiple pronunciations
                            result.push_back({dictionary_word, ordered_common_phonemes, "~"});
                        }
                    }
                }
            }
        }
        return result;
    }

    static inline std::vector<std::string> get_phonemes(const std::string& pronounciation)
    {
        return akil::string::split(pronounciation, ' ');
    }

    /**
     * @brief
     * Takes in two pronounciations: one from the word w, and the other
     * from the dictionary search(after), and returns the amount of phonemes
     * they have not only in common, but in order
     * @param query_phonemes
     * @param query_phonemes_count
     * @param dictionary_phonemes
     * @param dictionary_phonemes_count
     * @param order_matters
     * @return
     */
    static inline ushort get_num_of_common_phonemes(
        const std::vector<std::string>& query_phonemes,
        const ushort& query_phonemes_count,
        const std::vector<std::string>& dictionary_phonemes,
        const ushort& dictionary_phonemes_count,
        const bool order_matters = false)
    {
        ushort common_phonemes = 0;
        for(ushort query_phoneme_order = 0; query_phoneme_order < query_phonemes.size();
            ++query_phoneme_order)
        {
            const auto& query_phoneme = query_phonemes.at(query_phoneme_order);
            for(size_t dictionary_phoneme_order = 0;
                dictionary_phoneme_order < dictionary_phonemes.size();
                ++dictionary_phoneme_order)
            {
                if(order_matters)
                {
                    if(dictionary_phonemes.at(dictionary_phoneme_order) == query_phoneme)
                    {
                        if(dictionary_phonemes_count
                           == query_phonemes_count - query_phoneme_order)
                        {
                            //if phoneme is in tempAfter AND the amount of phonemes
                            //remaining in tempafter is the same as the amount in wPronounce
                            //where order is preserved, then add one to counter.
                            ++common_phonemes;
                        }
                    }
                }
                else
                {
                    if(dictionary_phonemes.at(dictionary_phoneme_order) == query_phoneme)
                    {
                        ++common_phonemes;
                    }
                }
            }
        }
        return common_phonemes;
    }

  private:
    std::unordered_map<std::string, std::tuple<std::string, std::vector<std::string>, ushort>>
        _phonemes_by_word;
    std::unordered_map<std::string, std::string> _soundex_encoding_by_word;
    std::unordered_map<std::string, std::pair<std::string, std::string>>
        _double_metaphone_encoding_by_word;
};

Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer(
    const std::filesystem::__cxx11::path& dictionary)
    : _impl(std::make_unique<
            Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer_Impl>(
        dictionary))
{}

Vocinity::Homophonic_Alternative_Composer::~Homophonic_Alternative_Composer()
{}

void
Vocinity::Homophonic_Alternative_Composer::set_in_memory_phonemes_dictionary(
    const std::unordered_map<std::string, std::string>& dictionary)
{
    _impl->set_in_memory_phonemes_dictionary(dictionary);
}

#ifdef SOUNDEX_AVAILABLE
void
Vocinity::Homophonic_Alternative_Composer::set_in_memory_soundex_dictionary(
    const std::unordered_map<std::string, std::string>& dictionary)
{
    _impl->set_in_memory_soundex_dictionary(dictionary);
}
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
void
Vocinity::Homophonic_Alternative_Composer::set_in_memory_double_metaphone_dictionary(
    const std::unordered_map<std::string, std::pair<std::string, std::string>>& dictionary)
{
    _impl->set_in_memory_double_metaphone_dictionary(dictionary);
}
#endif
Vocinity::Homophonic_Alternative_Composer::Alternative_Words_Of_Sentence
Vocinity::Homophonic_Alternative_Composer::get_alternatives(const std::string& reference,
                                                            const Instructions& instructions,
                                                            const bool parallel)
{
    Alternative_Words_Of_Sentence sentence_result;

    if(reference.empty())
    {
        return sentence_result;
    }

    std::string uppercase_reference;
    uppercase_reference.resize(reference.size());
    if(parallel)
    {
#ifdef CPP17_AVAILABLE
        std::transform(
            std::execution::par_unseq,
            reference.cbegin(),
            reference.cend(),
            uppercase_reference.begin(),
            [](const auto& c)
            { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); });
#else
        __gnu_parallel::transform(
            reference.cbegin(),
            reference.cend(),
            uppercase_reference.begin(),
            [](const auto& c)
            { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); });
#endif
    }
    else
    {
        std::transform(
            reference.cbegin(),
            reference.cend(),
            uppercase_reference.begin(),
            [](const auto& c)
            { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); });
    }

    const auto& words = akil::string::split(uppercase_reference, ' ');
    for(uint64_t order = 0; order < words.size(); ++order)
    {
        const auto& word = words.at(order);
        if(akil::memory::vector_contains(instructions.dismissed_word_indices, order))
        {
            sentence_result.push_back({{word, 0, "~"}});
            continue;
        }

        if(akil::memory::vector_contains(instructions.dismissed_words, word))
        {
            sentence_result.push_back({{word, 0, "~"}});
            continue;
        }

        sentence_result.emplace_back(_impl->get_alternatives(word, instructions, parallel));
    }

    return sentence_result;
}
