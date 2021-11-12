#include "Context_Scorer.hpp"

class Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer_Impl
{
  public:
    explicit Homophonic_Alternative_Composer_Impl(const std::filesystem::path& dictionary)
    {
        if(not dictionary.empty())
        {
            std::ifstream dictionary_file(dictionary);

            constexpr std::string_view delim{"  "};
            for(std::string line, word, pronounciation; std::getline(dictionary_file, line);)
            {
                if(akil::string::contains(line, ";;;"))
                {
                    continue;
                }

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
                              << line << "| is ignored" << std::endl;
                }
                const auto& phonemes    = get_phonemes(pronounciation);
                _phonemes_by_word[word] = {pronounciation, phonemes, phonemes.size()};
            }
        }
    }

    ~Homophonic_Alternative_Composer_Impl()
    {}

  public:
    void set_in_memory_phonemes_dictionary(
        const std::unordered_map<std::string, std::string>& dictionary)
    {
        _phonemes_by_word.clear();
        for(const auto& [word, pronounciation] : dictionary)
        {
            const auto& phonemes    = get_phonemes(pronounciation);
            _phonemes_by_word[word] = {pronounciation, phonemes, phonemes.size()};
        }
    }
#ifdef SOUNDEX_AVAILABLE
    void set_in_memory_soundex_dictionary(
        const std::unordered_map<std::string, std::string>& dictionary)
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
        if(_phonemes_by_word.empty() and _double_metaphone_encoding_by_word.empty()
           and _soundex_encoding_by_word.empty())
        {
            std::cout << "Dictionary is empty. Either you had to pass dict path to ctor or "
                         "set in-memory dictionary"
                      << std::endl;
            return result;
        }

        if(instructions.method == Matching_Method::Phoneme_Transcription
#ifdef LEVENSHTEIN_AVAILABLE
           or instructions.method == Matching_Method::Phoneme_Levenshtein
#endif

        )
        {
            bool exists = false;
            if(not _phonemes_by_word.empty())
            {
                exists = _phonemes_by_word.contains(query_word);
            }
            else
            {
                std::cout
                    << "Dictionary is empty. Either you had to pass dict path to ctor or "
                       "set in-memory dictionary"
                    << std::endl;
                return result;
            }

            if(exists)
            {
#ifdef LEVENSHTEIN_AVAILABLE
                if(instructions.method == Matching_Method::Phoneme_Levenshtein)
                {
                    result = get_alternatives_by_phoneme_levenshtein(query_word, instructions);
                }
                else
#endif
                {
                    result =
                        get_alternatives_by_phoneme_transcription(query_word, instructions);
                }
            }
            else
            {
//                std::cout << "Dictionary does not contain pronounciation for " << query_word
//                          << std::endl;
                return result;
            }
        }
#ifdef SOUNDEX_AVAILABLE
        else if(instructions.method == Matching_Method::Soundex)
        {
            result = get_alternatives_by_soundex(query_word);
        }
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
        else if(instructions.method == Matching_Method::Double_Metaphone)
        {
            result = get_alternatives_by_metaphone(query_word);
        }
#endif

        if(parallel)
        {
#ifdef CPP17_AVAILABLE
            std::sort(std::execution::par_unseq,
                      result.begin(),
                      result.end(),
                      [](const Alternative_Word& one, const Alternative_Word& another) -> bool
                      {
                          const auto& [_, first_distance, first_op]    = one;
                          const auto& [__, second_distance, second_op] = another;
                          const auto abs_first  = std::abs(first_distance);
                          const auto abs_second = std::abs(second_distance);
                          if(abs_first == abs_second)
                          {
                              if(first_op == "~" and (second_op not_eq "~"))
                              {
                                  return true;
                              }
                              else if((first_op not_eq "~") and (second_op not_eq "~"))
                              {
                                  if(first_op == "-" and second_op == "+")
                                  {
                                      return true;
                                  }
                              }
                          }
                          return abs_first < abs_second;
                      });
#else
            __gnu_parallel::sort(
                result.begin(),
                result.end(),
                [](const Alternative_Word& one, const Alternative_Word& another) -> bool
                {
                    const auto& [_, first_distance, first_op]    = one;
                    const auto& [__, second_distance, second_op] = another;
                    const auto abs_first                         = std::abs(first_distance);
                    const auto abs_second                        = std::abs(second_distance);
                    if(abs_first == abs_second)
                    {
                        if(first_op == "~" and (second_op not_eq "~"))
                        {
                            return true;
                        }
                        else if((first_op not_eq "~") and (second_op not_eq "~"))
                        {
                            if(first_op == "-" and second_op == "+")
                            {
                                return true;
                            }
                        }
                    }
                    return abs_first < abs_second;
                });
#endif
        }
        else
        {
            std::sort(result.begin(),
                      result.end(),
                      [](const Alternative_Word& one, const Alternative_Word& another) -> bool
                      {
                          const auto& [_, first_distance, first_op]    = one;
                          const auto& [__, second_distance, second_op] = another;
                          const auto abs_first  = std::abs(first_distance);
                          const auto abs_second = std::abs(second_distance);
                          if(abs_first == abs_second)
                          {
                              if(first_op == "~" and (second_op not_eq "~"))
                              {
                                  return true;
                              }
                              else if((first_op not_eq "~") and (second_op not_eq "~"))
                              {
                                  if(first_op == "-" and second_op == "+")
                                  {
                                      return true;
                                  }
                              }
                          }
                          return abs_first < abs_second;
                      });
        }

        if(instructions.max_best_num_alternatives)
        {
            if(result.size() > instructions.max_best_num_alternatives)
            {
                result.resize(instructions.max_best_num_alternatives);
            }
        }

        return result;
    }

  private:
#ifdef DOUBLE_METAPHONE_AVAILABLE
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives get_alternatives_by_metaphone(
        const std::string& query_word)
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;

        if(_double_metaphone_encoding_by_word.empty())
        {
            if(_soundex_encoding_by_word.empty())
            {
                if(_phonemes_by_word.empty())
                {
                    return {};
                }
                else
                {
                    for(const auto& [word, _] : _phonemes_by_word)
                    {
                        _double_metaphone_encoding_by_word[word] =
                            akil::string::get_double_metaphone_hash(word);
                    }
                }
            }
            else
            {
                for(const auto& [word, _] : _soundex_encoding_by_word)
                {
                    _double_metaphone_encoding_by_word[word] =
                        akil::string::get_double_metaphone_hash(word);
                }
            }
        }

        const auto& [query_primary_code, query_alternative_code] =
            _double_metaphone_encoding_by_word.contains(query_word)
                ? _double_metaphone_encoding_by_word.at(query_word)
                : akil::string::get_double_metaphone_hash(query_word);
        for(const auto& [dictionary_word, dictionary_codes] :
            _double_metaphone_encoding_by_word)
        {
            if(query_word == dictionary_word)
            {
                continue;
            }

            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
            if(has_parenthesis)
            {
                continue;
            }

            const auto& [dictionary_primary_code, dictionary_alternate_code] =
                dictionary_codes;
            if(dictionary_primary_code == query_primary_code)
            {
                result.push_back(
                    {dictionary_word,
                     std::abs((short) (query_word.size() - dictionary_word.size())),
                     "~"});
                continue;
            }
            if(dictionary_alternate_code == query_alternative_code)
            {
                result.push_back(
                    {dictionary_word,
                     std::abs((short) (query_word.size() - dictionary_word.size())),
                     "~"});
            }
        }
        return result;
    }
#endif
#ifdef SOUNDEX_AVAILABLE
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives get_alternatives_by_soundex(
        const std::string& query_word)
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;

        if(_soundex_encoding_by_word.empty())
        {
            if(_double_metaphone_encoding_by_word.empty())
            {
                if(_phonemes_by_word.empty())
                {
                    return {};
                }
                else
                {
                    for(const auto& [word, _] : _phonemes_by_word)
                    {
                        _soundex_encoding_by_word[word] = akil::string::get_soundex_hash(word);
                    }
                }
            }
            else
            {
                for(const auto& [word, _] : _double_metaphone_encoding_by_word)
                {
                    _soundex_encoding_by_word[word] = akil::string::get_soundex_hash(word);
                }
            }
        }

        const auto& query_code = _soundex_encoding_by_word.contains(query_word)
                                     ? _soundex_encoding_by_word.at(query_word)
                                     : akil::string::get_soundex_hash(query_word);
        for(const auto& [dictionary_word, dictionary_code] : _soundex_encoding_by_word)
        {
            if(query_word == dictionary_word)
            {
                continue;
            }

            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
            if(has_parenthesis)
            {
                continue;
            }

            if(dictionary_code == query_code)
            {
                result.push_back(
                    {dictionary_word,
                     std::abs((short) (query_word.size() - dictionary_word.size())),
                     "~"});
            }
        }

        return result;
    }
#endif

#ifdef LEVENSHTEIN_AVAILABLE
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives
    get_alternatives_by_phoneme_levenshtein(const std::string& query_word,
                                            const Instructions& instructions) const
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;
        const auto& [query_pronounciation, query_phonemes, query_phonemes_count] =
            _phonemes_by_word.at(query_word);

        const ushort max_distance =
            instructions.max_distance>-1 ? instructions.max_distance : query_phonemes_count;
        for(const auto& [dictionary_word, dictionary_pronounciation_info] : _phonemes_by_word)
        {
            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
            if(has_parenthesis)
            {
                continue;
            }

            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = dictionary_pronounciation_info;

            if(dictionary_word_pronounciation == query_pronounciation
               and dictionary_word not_eq query_word)
            {
                result.push_back({dictionary_word, 0, "~"});
                continue;
            }

            if(dictionary_word == query_word)
            {
                continue;
            }

            const auto& levenshtein_difference = akil::string::levenshtein_difference(
                query_pronounciation, dictionary_word_pronounciation);
            if(levenshtein_difference.size() <= max_distance)
            {
                ushort additions = 0, removals = 0, replacements = 0;
                for(const auto& ops : levenshtein_difference)
                {
                    if(ops.type == rapidfuzz::LevenshteinEditType::Insert)
                    {
                        ++additions;
                    }
                    else if(ops.type == rapidfuzz::LevenshteinEditType::Delete)
                    {
                        ++removals;
                    }
                    else if(ops.type == rapidfuzz::LevenshteinEditType::Replace)
                    {
                        ++replacements;
                    }
                }

                const std::string& final_op = additions > removals
                                                  ? (additions > replacements ? "+" : "~")
                                                  : (removals > replacements ? "-" : "~");
                result.push_back({dictionary_word, levenshtein_difference.size(), final_op});
            }
        }
        return result;
    }
#endif
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives
    get_alternatives_by_phoneme_transcription(const std::string& query_word,
                                              const Instructions& instructions) const
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;
        const auto& [query_pronounciation, query_phonemes, query_phonemes_count] =
            _phonemes_by_word.at(query_word);

        const ushort max_distance =
            instructions.max_distance>-1 ? instructions.max_distance : query_phonemes_count;
        for(const auto& [dictionary_word, dictionary_pronounciation_info] : _phonemes_by_word)
        {
            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
            if(has_parenthesis)
            {
                //ignores multiple pronunciations
                continue;
            }

            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = dictionary_pronounciation_info;

            if(dictionary_word_pronounciation == query_pronounciation
               and dictionary_word not_eq query_word)
            {
                result.push_back({dictionary_word, 0, "~"});
                continue;
            }

            const ushort num_of_common_phonemes =
                get_num_of_common_phonemes(query_phonemes,
                                           query_phonemes_count,
                                           dictionary_word_phonemes,
                                           dictionary_word_phonemes_count,
                                           false);

            for(short distance = 1; distance <= max_distance; ++distance)
            {
                //if the amount of similar Phonemes in w and the dictionary word is
                //equal to the the amount of phonemes in w, and the dictionary word has
                //one more phoneme total than w, add dictionary word to addPhoneme string
                if(num_of_common_phonemes == query_phonemes_count
                   and dictionary_word_phonemes_count == (query_phonemes_count + distance))
                {
                    result.push_back({dictionary_word, distance, "+"});
                }

                //if the amount of similar Phonemes in w and the dictionary word is
                //one less than the amount of phonemes in w, and the dictionary word has
                //one less total phonemes than w, add dictionary word to removePhoneme string
                else if(num_of_common_phonemes == query_phonemes_count - distance
                        and dictionary_word_phonemes_count
                                == (query_phonemes_count - distance))
                {
                    result.push_back({dictionary_word, distance, "-"});
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
                        result.push_back({dictionary_word,
                                          query_phonemes_count - ordered_common_phonemes,
                                          "~"});
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
        ushort common_phonemes                 = 0;
        ushort last_matched_dict_phoneme_order = 0;
        for(ushort query_phoneme_order = 0; query_phoneme_order < query_phonemes_count;
            ++query_phoneme_order)
        {
            const auto& query_phoneme = query_phonemes.at(query_phoneme_order);
            bool found                = false;
            for(size_t dictionary_phoneme_order = last_matched_dict_phoneme_order;
                dictionary_phoneme_order < dictionary_phonemes_count;
                ++dictionary_phoneme_order)
            {
                if(order_matters)
                {
                    if(dictionary_phonemes.at(dictionary_phoneme_order) == query_phoneme)
                    {
                        if(dictionary_phonemes_count - (dictionary_phoneme_order)
                           == query_phonemes_count - query_phoneme_order)
                        {
                            //if phoneme is in tempAfter AND the amount of phonemes
                            //remaining in tempafter is the same as the amount in wPronounce
                            //where order is preserved, then add one to counter.
                            ++common_phonemes;
                            last_matched_dict_phoneme_order = dictionary_phoneme_order;
                            found                           = true;
                        }
                    }
                }
                else
                {
                    if(dictionary_phonemes.at(dictionary_phoneme_order) == query_phoneme)
                    {
                        ++common_phonemes;
                        last_matched_dict_phoneme_order = dictionary_phoneme_order;
                        found                           = true;
                    }
                }
                if(found)
                {
                    break;
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
