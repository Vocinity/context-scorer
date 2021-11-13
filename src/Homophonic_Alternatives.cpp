#include "Context_Scorer.hpp"

class Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer_Impl
{
  public:
    explicit Homophonic_Alternative_Composer_Impl(const std::filesystem::path& dictionary,
                                                  const Use_More mode)
        : _mode(mode)
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

                if(akil::string::contains(line, ")"))
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
                              << " is corrupted, 2 items expected in each line,\n entry of: |"
                              << line << "| is ignored" << std::endl;
                }
                const auto& phonemes    = get_phonemes(pronounciation);
                _phonemes_by_word[word] = {pronounciation, phonemes, phonemes.size()};
            }

            if(_mode == Use_More::And_More_Memory)
            {
                for(const auto& [query_word, outer_pron] : _phonemes_by_word)
                {
                    const auto& [query_pronounciation, query_phonemes, query_phonemes_count] =
                        outer_pron;

#ifdef SOUNDEX_AVAILABLE
                    const auto soundex_encoding = akil::string::get_soundex_hash(query_word);
                    _soundex_encoding_by_word[query_word] = soundex_encoding;
                    _word_by_soundex_encoding[soundex_encoding].push_back(query_word);
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
                    const auto double_metaphone_encoding =
                        akil::string::get_double_metaphone_hash(query_word);
                    _double_metaphone_encoding_by_word[query_word] = double_metaphone_encoding;
                    _word_by_metaphone_encoding[double_metaphone_encoding.first].push_back(
                        query_word);
                    _word_by_metaphone_encoding[double_metaphone_encoding.second].push_back(
                        query_word);
#endif
                    const ushort max_distance = query_phonemes_count;
                    _phoneme_index[query_word][Matching_Method::Phoneme_Transcription].resize(
                        query_phonemes_count + 1);
                    _phoneme_index[query_word][Matching_Method::Phoneme_Levenshtein].resize(
                        query_phonemes_count + 1);

                    for(const auto& [dictionary_word, inner_pron] : _phonemes_by_word)
                    {
                        const auto& [dictionary_word_pronounciation,
                                     dictionary_word_phonemes,
                                     dictionary_word_phonemes_count] = inner_pron;

                        if(dictionary_word_phonemes_count == query_phonemes_count)
                        {
                            if(not strcmp(dictionary_word_pronounciation.c_str(),
                                          query_pronounciation.c_str()))
                            {
                                if(strcmp(dictionary_word.c_str(), query_word.c_str()))
                                {
                                    _phoneme_index[query_word]
                                                  [Matching_Method::Phoneme_Transcription][0]
                                                      .push_back({dictionary_word, 0, "~"});
                                    _phoneme_index[query_word]
                                                  [Matching_Method::Phoneme_Levenshtein][0]
                                                      .push_back({dictionary_word, 0, "~"});
                                    continue;
                                }
                            }
                        }

                        const auto& levenshtein_difference =
                            akil::string::levenshtein_difference(
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

                            const std::string& final_op =
                                additions > removals ? (additions > replacements ? "+" : "~")
                                                     : (removals > replacements ? "-" : "~");
                            _phoneme_index[query_word][Matching_Method::Phoneme_Levenshtein][0]
                                .push_back({dictionary_word,
                                            levenshtein_difference.size(),
                                            final_op});
                        }


                        const ushort num_of_common_phonemes =
                            get_num_of_common_phonemes(query_phonemes,
                                                       query_phonemes_count,
                                                       dictionary_word_phonemes,
                                                       dictionary_word_phonemes_count,
                                                       false);

                        for(ushort distance = 1; distance <= max_distance; ++distance)
                        {
                            if(num_of_common_phonemes == (query_phonemes_count - distance))
                            {
                                if(dictionary_word_phonemes_count
                                   == query_phonemes_count - distance)
                                {
                                    _phoneme_index[query_word]
                                                  [Matching_Method::Phoneme_Transcription]
                                                  [distance]
                                                      .push_back(
                                                          {dictionary_word, distance, "-"});
                                }
                                else if(dictionary_word_phonemes_count == query_phonemes_count)
                                {
                                    const auto ordered_common_phonemes =
                                        get_num_of_common_phonemes(
                                            query_phonemes,
                                            query_phonemes_count,
                                            dictionary_word_phonemes,
                                            dictionary_word_phonemes_count,
                                            true);
                                    if(ordered_common_phonemes
                                       == (query_phonemes_count - distance))
                                    {
                                        _phoneme_index[query_word]
                                                      [Matching_Method::Phoneme_Transcription]
                                                          .at(query_phonemes_count
                                                              - ordered_common_phonemes)
                                                          .push_back(
                                                              {dictionary_word,
                                                               query_phonemes_count
                                                                   - ordered_common_phonemes,
                                                               "~"});
                                    }
                                }
                            }
                            else if(num_of_common_phonemes == query_phonemes_count)
                            {
                                if(dictionary_word_phonemes_count
                                   == (query_phonemes_count + distance))
                                {
                                    _phoneme_index[query_word]
                                                  [Matching_Method::Phoneme_Transcription]
                                                  [distance]
                                                      .push_back(
                                                          {dictionary_word, distance, "+"});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ~Homophonic_Alternative_Composer_Impl()
    {}

  public:
    void set_in_memory_phonemes_dictionary(
        const Unordered_Map<std::string, std::string>& dictionary)
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
        const Unordered_Map<std::string, std::string>& dictionary)
    {
        _soundex_encoding_by_word = dictionary;
    }
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
    void set_in_memory_double_metaphone_dictionary(
        const Unordered_Map<std::string, std::pair<std::string, std::string>>& dictionary)
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

#ifdef CPP17_AVAILABLE
        std::sort(std::execution::unseq,
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
#else
        std::sort(result.begin(),
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

        if(_mode == Use_More::And_More_Memory)
        {
            if(_word_by_metaphone_encoding.contains(query_primary_code))
            {
                const auto& matches = _word_by_metaphone_encoding[query_primary_code];
                for(const auto& match : matches)
                {
                    if(not strcmp(match.c_str(), query_word.c_str()))
                    {
                        continue;
                    }
                    result.push_back({match, (query_word.size() - match.size()), "~"});
                }
            }
            else if(_word_by_metaphone_encoding.contains(query_alternative_code))
            {
                const auto& matches = _word_by_metaphone_encoding[query_alternative_code];
                for(const auto& match : matches)
                {
                    if(not strcmp(match.c_str(), query_word.c_str()))
                    {
                        continue;
                    }
                    result.push_back({match, (query_word.size() - match.size()), "~"});
                }
            }

            return result;
        }

        for(const auto& [dictionary_word, dictionary_codes] :
            _double_metaphone_encoding_by_word)
        {
            //            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
            //            if(has_parenthesis)
            //            {
            //                continue;
            //            }

            const auto& [dictionary_primary_code, dictionary_alternate_code] =
                dictionary_codes;
            if(not strcmp(dictionary_primary_code.c_str(), query_primary_code.c_str()))
            {
                if(not strcmp(query_word.c_str(), dictionary_word.c_str()))
                {
                    continue;
                }

                result.push_back(
                    {dictionary_word, (query_word.size() - dictionary_word.size()), "~"});
                continue;
            }
            if(not strcmp(dictionary_alternate_code.c_str(), query_alternative_code.c_str()))
            {
                if(not strcmp(query_word.c_str(), dictionary_word.c_str()))
                {
                    continue;
                }

                result.push_back(
                    {dictionary_word, (query_word.size() - dictionary_word.size()), "~"});
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

        if(_mode == Use_More::And_More_Memory)
        {
            if(_word_by_soundex_encoding.contains(query_code))
            {
                const auto& matches = _word_by_soundex_encoding[query_code];
                for(const auto& match : matches)
                {
                    if(not strcmp(match.c_str(), query_word.c_str()))
                    {
                        continue;
                    }
                    result.push_back({match, (query_word.size() - match.size()), "~"});
                }
            }

            return result;
        }


        for(const auto& [dictionary_word, dictionary_code] : _soundex_encoding_by_word)
        {
            //            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
            //            if(has_parenthesis)
            //            {
            //                continue;
            //            }

            if(not strcmp(dictionary_code.c_str(), query_code.c_str()))
            {
                if(not strcmp(query_word.c_str(), dictionary_word.c_str()))
                {
                    continue;
                }

                result.push_back(
                    {dictionary_word,
                     0 /*std::abs((short) (query_word.size() - dictionary_word.size()))*/,
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
            instructions.max_distance > -1 ? instructions.max_distance : query_phonemes_count;

        if(_mode == Use_More::And_More_Memory)
        {
            for(ushort distance = 0; distance < max_distance; ++distance)
            {
                if(_phoneme_index.at(query_word)
                       .at(Matching_Method::Phoneme_Levenshtein)
                       .size()
                   > distance)
                {
                    const auto& distanced_items = _phoneme_index.at(query_word)
                                                      .at(Matching_Method::Phoneme_Levenshtein)
                                                      .at(distance);
                    result.insert(std::end(result),
                                  std::begin(distanced_items),
                                  std::end(distanced_items));
                }
            }
            return result;
        }

        for(const auto& [dictionary_word, dictionary_pronounciation_info] : _phonemes_by_word)
        {
            //            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
            //            if(has_parenthesis)
            //            {
            //                continue;
            //            }
            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = dictionary_pronounciation_info;

            if(not strcmp(dictionary_word_pronounciation.c_str(),
                          query_pronounciation.c_str()))
            {
                if(not strcmp(dictionary_word.c_str(), query_word.c_str()))
                {
                    continue;
                }

                result.push_back({dictionary_word, 0, "~"});
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
        auto chrono = std::chrono::high_resolution_clock::now();

        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;
        const auto& [query_pronounciation, query_phonemes, query_phonemes_count] =
            _phonemes_by_word.at(query_word);

        const ushort max_distance =
            instructions.max_distance > -1 ? instructions.max_distance : query_phonemes_count;

        if(_mode == Use_More::And_More_Memory)
        {
            for(ushort distance = 0; distance < max_distance; ++distance)
            {
                if(_phoneme_index.at(query_word)
                       .at(Matching_Method::Phoneme_Transcription)
                       .size()
                   > distance)
                {
                    const auto& distanced_items =
                        _phoneme_index.at(query_word)
                            .at(Matching_Method::Phoneme_Transcription)
                            .at(distance);
                    result.insert(std::end(result),
                                  std::begin(distanced_items),
                                  std::end(distanced_items));
                }
            }
            return result;
        }

        for(const auto& [dictionary_word, dictionary_pronounciation_info] : _phonemes_by_word)
        {
            //            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
            //            if(has_parenthesis)
            //            {
            //                //ignores multiple pronunciations
            //                continue;
            //            }

            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = dictionary_pronounciation_info;
            if(dictionary_word_phonemes_count == query_phonemes_count)
            {
                if(not strcmp(dictionary_word_pronounciation.c_str(),
                              query_pronounciation.c_str()))
                {
                    if(strcmp(dictionary_word.c_str(), query_word.c_str()))
                    {
                        result.push_back({dictionary_word, 0, "~"});
                        continue;
                    }
                }
            }

            const ushort num_of_common_phonemes =
                get_num_of_common_phonemes(query_phonemes,
                                           query_phonemes_count,
                                           dictionary_word_phonemes,
                                           dictionary_word_phonemes_count,
                                           false);

            for(ushort distance = 1; distance <= max_distance; ++distance)
            {
                if(num_of_common_phonemes == (query_phonemes_count - distance))
                {
                    if(dictionary_word_phonemes_count == query_phonemes_count - distance)
                    {
                        result.push_back({dictionary_word, distance, "-"});
                    }
                    else if(dictionary_word_phonemes_count == query_phonemes_count)
                    {
                        const auto ordered_common_phonemes =
                            get_num_of_common_phonemes(query_phonemes,
                                                       query_phonemes_count,
                                                       dictionary_word_phonemes,
                                                       dictionary_word_phonemes_count,
                                                       true);
                        if(ordered_common_phonemes == (query_phonemes_count - distance))
                        {
                            result.push_back({dictionary_word,
                                              query_phonemes_count - ordered_common_phonemes,
                                              "~"});
                        }
                    }
                }
                else if(num_of_common_phonemes == query_phonemes_count)
                {
                    if(dictionary_word_phonemes_count == (query_phonemes_count + distance))
                    {
                        result.push_back({dictionary_word, distance, "+"});
                    }
                }
            }
        }

        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - chrono)
                                  .count();

        std::cout << "time_past_inside transcription_processor phonemes: " << duration
                  << std::endl;
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

        if(order_matters)
        {
            ushort last_matched_dict_phoneme_order = 0;
            for(ushort query_phoneme_order = 0; query_phoneme_order < query_phonemes_count;
                ++query_phoneme_order)
            {
                const auto& query_phoneme = query_phonemes[query_phoneme_order];

                for(ushort dictionary_phoneme_order = last_matched_dict_phoneme_order;
                    dictionary_phoneme_order < dictionary_phonemes_count;
                    ++dictionary_phoneme_order)
                {
                    const auto& dictionary_phoneme =
                        dictionary_phonemes[dictionary_phoneme_order];

                    if(dictionary_phoneme.size() == query_phoneme.size())
                    {
                        if(not strcmp(dictionary_phonemes[dictionary_phoneme_order].c_str(),
                                      query_phoneme.c_str()))
                        {
                            if(dictionary_phonemes_count - (dictionary_phoneme_order)
                               == query_phonemes_count - query_phoneme_order)
                            {
                                //if phoneme is in tempAfter AND the amount of phonemes
                                //remaining in tempafter is the same as the amount in wPronounce
                                //where order is preserved, then add one to counter.
                                ++common_phonemes;
                                last_matched_dict_phoneme_order = dictionary_phoneme_order;
                                break;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for(const auto& query_phoneme : query_phonemes)
            {
                for(const auto& dictionary_phoneme : dictionary_phonemes)
                {
                    if(dictionary_phoneme.size() == query_phoneme.size())
                    {
                        if(not strcmp(dictionary_phoneme.c_str(), query_phoneme.c_str()))
                        {
                            ++common_phonemes;
                            break;
                        }
                    }
                }
            }
        }

        return common_phonemes;
    }

  private:
    Use_More _mode = Use_More::And_More_Memory;
    Unordered_Map<std::string, std::tuple<std::string, std::vector<std::string>, ushort>>
        _phonemes_by_word;
    Unordered_Map<std::string, Unordered_Map<Matching_Method, std::vector<Word_Alternatives>>>
        _phoneme_index;
#ifdef SOUNDEX_AVAILABLE
    Unordered_Map<std::string, std::vector<std::string>> _word_by_soundex_encoding;
    Unordered_Map<std::string, std::string> _soundex_encoding_by_word;
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
    Unordered_Map<std::string, std::vector<std::string>> _word_by_metaphone_encoding;
    Unordered_Map<std::string, std::pair<std::string, std::string>>
        _double_metaphone_encoding_by_word;
#endif
};

Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer(
    const Use_More mode,
    const std::filesystem::__cxx11::path& dictionary)
    : _impl(std::make_unique<
            Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer_Impl>(
        dictionary,
        mode))
{}

Vocinity::Homophonic_Alternative_Composer::~Homophonic_Alternative_Composer()
{}

void
Vocinity::Homophonic_Alternative_Composer::set_in_memory_phonemes_dictionary(
    const Unordered_Map<std::string, std::string>& dictionary)
{
    _impl->set_in_memory_phonemes_dictionary(dictionary);
}

#ifdef SOUNDEX_AVAILABLE
void
Vocinity::Homophonic_Alternative_Composer::set_in_memory_soundex_dictionary(
    const Unordered_Map<std::string, std::string>& dictionary)
{
    _impl->set_in_memory_soundex_dictionary(dictionary);
}
#endif
#ifdef DOUBLE_METAPHONE_AVAILABLE
void
Vocinity::Homophonic_Alternative_Composer::set_in_memory_double_metaphone_dictionary(
    const Unordered_Map<std::string, std::pair<std::string, std::string>>& dictionary)
{
    _impl->set_in_memory_double_metaphone_dictionary(dictionary);
}
#endif
Vocinity::Homophonic_Alternative_Composer::Alternative_Words_Of_Sentence
Vocinity::Homophonic_Alternative_Composer::get_alternatives(const std::string& reference,
                                                            const Instructions& instructions,
                                                            const bool parallel)
{
    auto chrono = std::chrono::high_resolution_clock::now();

    Alternative_Words_Of_Sentence sentence_result;

    if(reference.empty())
    {
        return sentence_result;
    }

    std::string uppercase_reference;
    uppercase_reference.resize(reference.size());

#ifdef CPP17_AVAILABLE
    std::transform(std::execution::unseq,
                   reference.cbegin(),
                   reference.cend(),
                   uppercase_reference.begin(),
                   [](const auto& c)
                   { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); });
#else
    std::transform(reference.cbegin(),
                   reference.cend(),
                   uppercase_reference.begin(),
                   [](const auto& c)
                   { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); });
#endif


    const auto& words = akil::string::split(uppercase_reference, ' ');
    for(ushort order = 0; order < words.size(); ++order)
    {
        const auto& word = words.at(order);
        if(akil::memory::vector_contains(instructions.dismissed_word_indices, order))
        {
            sentence_result.push_back({{}});
            continue;
        }

        if(akil::memory::vector_contains(instructions.dismissed_words, word))
        {
            sentence_result.push_back({{}});
            continue;
        }

        sentence_result.emplace_back(_impl->get_alternatives(word, instructions, parallel));
    }

    const auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::high_resolution_clock::now() - chrono)
                                    .count();
    std::cout << "time spent in total: " << total_duration << std::endl;

    return sentence_result;
}
