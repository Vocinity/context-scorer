#include "Homophonic-Alternatives.hpp"

#include <QFile>
class Phoneme_Synthesizer
{
    template <typename Token_ID = int64_t>
    class Phoneme_Synth_Tokenizer
    {
      public:
        Phoneme_Synth_Tokenizer()
        {
            static std::mutex mutex;
            const std::lock_guard lock(mutex);
            if(_word_to_id_dictionary.empty())
            {
                // clang-format off
                _word_to_id_dictionary={
                    {"&apos;",26},{"-",31},{".",32},{"</s>",2},{"<pad>",1},{"<s>",0},
                    {"<unk>",3},{"a",5},{"b",20},{"c",13},{"d",14},{"e",4},{"f",23},{"g",18},{"h",17},
                    {"i",7},{"j",28},{"k",21},{"l",12},{"m",15},{"madeupword0000",33},{"madeupword0001",34},
                    {"madeupword0002",35},{"madeupword0003",36},{"madeupword0004",37},{"madeupword0005",38},
                    {"madeupword0006",39},{"n",9},{"o",10},{"p",19},{"q",30},{"r",6},{"s",8},{"t",11},{"u",16},
                    {"v",25},{"w",24},{"x",29},{"y",22},{"z",27}};
                for(const auto& [word,id]:_word_to_id_dictionary)
                {
                    _id_to_word_dictionary[id]=word;
                }

                _phoneme_to_id_dictionary={
                    {"</s>",2},{"<pad>",1},{"<s>",0},{"<unk>",3},{"AA0",40},{"AA1",20},{"AA2",47},{"AE0",56},
                    {"AE1",21},{"AE2",46},{"AH0",4},{"AH1",35},{"AH2",61},{"AO0",58},{"AO1",34},{"AO2",55},
                    {"AW0",69},{"AW1",53},{"AW2",64},{"AY0",59},{"AY1",36},{"AY2",48},{"B",17},{"CH",41},{"D",11},
                    {"DH",66},{"EH0",49},{"EH1",16},{"EH2",44},{"ER0",15},{"ER1",42},{"ER2",65},{"EY0",62},{"EY1",29},
                    {"EY2",45},{"F",23},{"G",24},{"HH",28},{"IH0",12},{"IH1",22},{"IH2",43},{"IY0",16},{"IY1",26},
                    {"IY2",51},{"JH",38},{"K",10},{"L",7},{"M",13},{"madeupword0000",73},{"madeupword0001",74},
                    {"madeupword0002",75},{"madeupword0003",76},{"madeupword0004",77},{"madeupword0005",78},
                    {"madeupword0006",79},{"N",5},{"NG",27},{"OW0",33},{"OW1",32},{"OW2",52},{"OY0",72},{"OY1",63},
                    {"OY2",71},{"P",19},{"R",9},{"S",6},{"SH",31},{"T",8},{"TH",50},{"UH0",70},{"UH1",57},
                    {"UH2",68},{"UW0",54},{"UW1",37},{"UW2",60},{"V",25},{"W",30},{"Y",39},{"Z",14},{"ZH",67}
                };
                for(const auto& [phoneme,id]:_phoneme_to_id_dictionary)
                {
                    _id_to_phoneme_dictionary[id]=phoneme;
                }
                // clang-format on
            }
        }
        torch::Tensor encode_word(const std::string& word,
                                  const bool drop_padding = true) const
        {
            std::vector<Token_ID> ids;
            ids.resize(word.size());
            std::transform(std::execution::unseq,
                           word.cbegin(),
                           word.cend(),
                           ids.begin(),
                           [](const auto& c)
                           {
                               const auto& c_lower = static_cast<char>(
                                   std::tolower(static_cast<unsigned char>(c)));
                               if(not _word_to_id_dictionary.contains(std::string{c_lower}))
                               {
                                   return _word_to_id_dictionary.at("<unk>");
                               }
                               return _word_to_id_dictionary.at(std::string{c_lower});
                           });
            if(drop_padding)
            {
                akil::memory::vector_erase_remove(ids, _phoneme_to_id_dictionary.at("<pad>"));
            }
            ids << _word_to_id_dictionary.at("</s>");
            return akil::memory::vector_1d_to_tensor_1d_by_copy<Token_ID>(ids)
                .to(torch::kLong)
                .unsqueeze(0);
        }
        torch::Tensor encode_phoneme(const std::string& phoneme_sequence,
                                     const bool drop_padding = true) const
        {
            std::vector<Token_ID> ids;
            const std::vector<std::string>& phonemes =
                akil::string::split(phoneme_sequence, ' ');
            ids.resize(phonemes.size());
            std::transform(std::execution::unseq,
                           phonemes.cbegin(),
                           phonemes.cend(),
                           ids.begin(),
                           [](const auto& c)
                           {
                               if(not _phoneme_to_id_dictionary.contains(c))
                               {
                                   return _phoneme_to_id_dictionary.at("<unk>");
                               }
                               return _phoneme_to_id_dictionary.at(c);
                           });
            if(drop_padding)
            {
                akil::memory::vector_erase_remove(ids, _phoneme_to_id_dictionary.at("<pad>"));
            }
            ids << _phoneme_to_id_dictionary.at("</s>");
            return akil::memory::vector_1d_to_tensor_1d_by_copy<Token_ID>(ids)
                .to(torch::kLong)
                .unsqueeze(0);
        }
        std::string decode_word(const torch::Tensor& ids, const bool drop_padding = true) const
        {
            std::string chars;
            const auto id_vec = akil::memory::tensor_1d_to_span_1d_no_copy<Token_ID>(ids);
            chars.resize(id_vec.size());
            for(const auto& c : id_vec)
            {
                if(not _id_to_word_dictionary.contains(c))
                {
                    continue;
                }
                if(c == _word_to_id_dictionary.at("<s>")
                   or c == _word_to_id_dictionary.at("</s>"))
                {
                    continue;
                }
                chars += _id_to_word_dictionary.at(c);
            }
            if(drop_padding)
            {
                akil::string::find_replace(chars, "<pad>", "");
            }
            return chars;
        }

        std::string decode_phoneme(const torch::Tensor& ids,
                                   const bool drop_padding = true) const
        {
            std::string chars;
            const auto id_vec = akil::memory::tensor_1d_to_span_1d_no_copy<Token_ID>(ids);
            for(const auto& c : id_vec)
            {
                if(not _id_to_phoneme_dictionary.contains(c))
                {
                    continue;
                }
                if(c == _phoneme_to_id_dictionary.at("<s>")
                   or c == _phoneme_to_id_dictionary.at("</s>"))
                {
                    continue;
                }
                chars += _id_to_phoneme_dictionary.at(c) + " ";
            }
            chars.resize(chars.size() - 1);
            if(drop_padding)
            {
                akil::string::find_replace(chars, "<pad>", "");
            }
            return chars;
        }

      private:
        static inline Unordered_Map<std::string, Token_ID> _word_to_id_dictionary;
        static inline Unordered_Map<std::string, Token_ID> _phoneme_to_id_dictionary;
        static inline Unordered_Map<Token_ID, std::string> _id_to_word_dictionary;
        static inline Unordered_Map<Token_ID, std::string> _id_to_phoneme_dictionary;
    };

    struct Generated_Phoneme
    {
        Generated_Phoneme() = default;
        Generated_Phoneme(const c10::Dict<torch::jit::IValue, torch::jit::IValue>& candidate)
        {
            for(const auto& field : candidate)
            {
                if(field.key() == "tokens")
                {
                    tokens = field.value().toTensor().detach();
                }
                else if(field.key() == "score")
                {
                    score = field.value().toTensor().item().toDouble();
                }
                else if(field.key() == "positional_scores")
                {
                    positional_scores = field.value().toTensor().detach();
                }
            }
        }
        torch::Tensor tokens;
        torch::Tensor positional_scores;
        double score = 0;
    };

  public:
    Phoneme_Synthesizer()
    {
        QFile qFile(":/assets/transformer_g2p-cpu.jit");
        if(qFile.open(QFile::ReadOnly))
        {
            const auto data = qFile.readAll();
            if(data.isEmpty())
            {
                throw std::runtime_error("vad_model cpu-cuda_8K-16K.jit is corrupted");
            }

            std::istringstream iss(data.toStdString());
            _phoneme_synthesizer_model = torch::jit::load(iss);
            _phoneme_synthesizer_model.to(torch::kCPU);
            _phoneme_synthesizer_model.eval();
        }
        else
        {
            throw std::runtime_error("vad torch jit model is corrupted");
        }
    }

    std::string get_phonemes(const std::string& word)
    {
        std::string preprocessed = word;
        if(akil::string::contains(word, " "))
        {
            std::cout << "You should send single word to Phoneme_Synthesizer::get_phonemes, "
                         "dropping spaces and treating as single word";
            akil::string::find_replace(preprocessed, " ", "");
        }

        const auto tokenized_word = _tokenizer.encode_word(preprocessed);

        const auto phoneme_ids =
            torch::jit::IValue(_phoneme_synthesizer_model
                                   .forward(std::vector<torch::jit::IValue>{
                                       torch::jit::IValue(tokenized_word)})
                                   .toList()
                                   .get(0))
                .toList();
        std::vector<Generated_Phoneme> synthetic_ones;
        for(const auto& cand : phoneme_ids)
        {
            const auto& fields = torch::jit::IValue(cand).toGenericDict();
            synthetic_ones << Generated_Phoneme(fields);
        }

        //    std::sort(std::execution::unseq,
        //              synthetic_ones.begin(),
        //              synthetic_ones.end(),
        //              [](const auto& one, const auto& another) -> bool
        //              {
        //                  return one.score > another.score;
        //              });

        return _tokenizer.decode_phoneme(synthetic_ones.front().tokens);
    }

  private:
    Phoneme_Synth_Tokenizer<> _tokenizer;
    torch::jit::Module _phoneme_synthesizer_model;
};


class Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer_Impl
{
    enum class Use_More : short {
        /** Pre-index whole cartesian relations and just lookup in runtime. */
        And_More_Memory = 0,
        /**  Compute again and again.*/
        CPU = 1
    };

  public:
#ifdef SIG_SLOT_AVAILABLE
    /**async reports*/
  public:
    sigslot::signal<std::string, std::vector<std::string>> saw_strange_word;
#endif

  public:
    explicit Homophonic_Alternative_Composer_Impl(
        const std::vector<std::pair<std::string, std::string>>& phonetics_dictionary)
        : _mode(Use_More::CPU)
    {
        for(const auto& [word, pronounciation] : phonetics_dictionary)
        {
            const auto& phonemes     = get_phonemes(pronounciation);
            const auto phonemes_size = phonemes.size();
            _phonemes_by_word[word]  = _phonemes_vector.size();
            _phonemes_vector.push_back({word, {pronounciation, phonemes, phonemes_size}});
        }
    }

    ~Homophonic_Alternative_Composer_Impl()
    {}

  public:
    void set_precomputed_phoneme_similarity_map(
        std::vector<std::vector<std::vector<std::pair<size_t, char>>>> map,
        const bool levenshtein)
    {
        _phoneme_index        = std::move(map);
        _mode                 = Use_More::And_More_Memory;
        _is_levensthein_index = levenshtein;
    }

  public:
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives get_alternatives(
        const std::string& query_word,
        const Instructions& instructions,
        const bool parallel)
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;
        if(_phonemes_vector.empty())
        {
            std::cout << "Dictionary is empty. Either you had to pass dict path to ctor or "
                         "set in-memory dictionary."
                      << std::endl;
            return result;
        }

#ifdef LEVENSHTEIN_AVAILABLE
        if(instructions.method == Matching_Method::Phoneme_Levenshtein)
        {
            result = get_alternatives_by_phoneme_levenshtein(query_word, instructions);
        }
        else
#endif
        {
            result = get_alternatives_by_phoneme_transcription(query_word, instructions);
        }


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

        if(instructions.max_num_of_best_homophonic_alternatives)
        {
            if(result.size() > instructions.max_num_of_best_homophonic_alternatives)
            {
                result.resize(instructions.max_num_of_best_homophonic_alternatives);
            }
        }

        return result;
    }

  private:
#ifdef LEVENSHTEIN_AVAILABLE
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives
    get_alternatives_by_phoneme_levenshtein(const std::string& query_word,
                                            const Instructions& instructions)
    {
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;

        ushort query_phonemes_count = 0;
        std::string query_pronounciation;
        std::vector<std::string> query_phonemes;
        const bool synthetic                 = not _phonemes_by_word.contains(query_word);
        int32_t homonym_index_of_madeup_word = -1;

        if(not synthetic)
        {
            const auto& [dict_query_pronounciation,
                         dict_query_phonemes,
                         dict_query_phonemes_count] =
                _phonemes_vector.at(_phonemes_by_word.at(query_word)).second;
            query_pronounciation = dict_query_pronounciation;
            query_phonemes       = dict_query_phonemes;
            query_phonemes_count = dict_query_phonemes_count;
        }
        else
        {
            query_pronounciation = _phoneme_synthesizer.get_phonemes(query_word);
            query_phonemes       = get_phonemes(query_pronounciation);
            query_phonemes_count = query_phonemes.size();

#	ifdef SIG_SLOT_AVAILABLE
            saw_strange_word(query_word, query_phonemes);
#	endif

            for(const auto& [word, phonemes] : _phonemes_vector)
            {
                auto [dict_query_pronounciation,
                      dict_query_phonemes,
                      dict_query_phonemes_count] = phonemes;
                //TODO: delete find_and_replace logic and have const auto& after fixing \r endings of dictionary by getting a fresh dump
                akil::string::find_replace(dict_query_pronounciation, "\r", "");
                akil::string::find_replace(dict_query_pronounciation, "\n", "");
                if(akil::string::are_equal(dict_query_pronounciation, query_pronounciation))
                {
                    homonym_index_of_madeup_word = _phonemes_by_word.at(word);
                    break;
                }
            }
        }

        if(_mode == Use_More::And_More_Memory and _is_levensthein_index
           and (synthetic ? homonym_index_of_madeup_word > -1 : true))
        {
            const auto query_word_indice =
                (synthetic ? homonym_index_of_madeup_word : _phonemes_by_word.at(query_word));
            const auto& distanced_items = _phoneme_index.at(query_word_indice);
            const ushort max_distance =
                instructions.max_distance > -1
                    ? std::min((size_t) instructions.max_distance, distanced_items.size())
                    : distanced_items.size();
            for(ushort distance = 0; distance <= max_distance; ++distance)
            {
                const auto& items_of_distance = distanced_items.at(distance);
                const auto result_range_begin = result.size();
                result.resize(result_range_begin + items_of_distance.size());
                std::transform(std::execution::unseq,
                               items_of_distance.cbegin(),
                               items_of_distance.cend(),
                               result.begin() + result_range_begin,
                               [&](const std::pair<size_t, char>& item) -> Alternative_Word
                               {
                                   return {
                                       _phonemes_vector.at(item.first).first,
                                       distance,
                                       (item.second ? (item.second > 0 ? "+" : "-") : "~")};
                               });
            }
            return result;
        }

        if(_mode == Use_More::And_More_Memory and _is_levensthein_index
           and (synthetic and homonym_index_of_madeup_word == -1))
        {
            std::cout << "OOV procedure. Walking looong way now." << std::endl;
        }

        const ushort max_distance =
            instructions.max_distance > -1 ? instructions.max_distance : query_phonemes_count;

        std::mutex mutex;
        auto processor = [&](const std::string& dictionary_word,
                             const std::tuple<std::string, std::vector<std::string>, ushort>&
                                 dictionary_pronounciation_info)
        {
            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = dictionary_pronounciation_info;

            if(akil::string::are_equal(dictionary_word_pronounciation, query_pronounciation))
            {
                if(akil::string::are_equal(dictionary_word, query_word))
                {
                    return;
                }
                const std::lock_guard lock(mutex);
                result.push_back({dictionary_word, 0, "~"});
                return;
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
                const std::lock_guard lock(mutex);
                result.push_back({dictionary_word, levenshtein_difference.size(), final_op});
            }
        };

        std::for_each(std::execution::par_unseq,
                      _phonemes_vector.cbegin(),
                      _phonemes_vector.cend(),
                      [&](const auto& item)
                      {
                          const auto& [dictionary_word, dictionary_pronounciation_info] = item;

                          //            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
                          //            if(has_parenthesis)
                          //            {
                          //                continue;
                          //            }

                          processor(dictionary_word, dictionary_pronounciation_info);
                      });

        return result;
    }
#endif
    Vocinity::Homophonic_Alternative_Composer::Word_Alternatives
    get_alternatives_by_phoneme_transcription(const std::string& query_word,
                                              const Instructions& instructions)
    {
#ifdef PROFILE_TIMING
        auto chrono = std::chrono::high_resolution_clock::now();
#endif
        Vocinity::Homophonic_Alternative_Composer::Word_Alternatives result;

        ushort query_phonemes_count = 0;
        std::string query_pronounciation;
        std::vector<std::string> query_phonemes;
        const bool synthetic                 = not _phonemes_by_word.contains(query_word);
        int32_t homonym_index_of_madeup_word = -1;

        if(not synthetic)
        {
            const auto& [dict_query_pronounciation,
                         dict_query_phonemes,
                         dict_query_phonemes_count] =
                _phonemes_vector.at(_phonemes_by_word.at(query_word)).second;
            query_pronounciation = dict_query_pronounciation;
            query_phonemes       = dict_query_phonemes;
            query_phonemes_count = dict_query_phonemes_count;
        }
        else
        {
            query_pronounciation = _phoneme_synthesizer.get_phonemes(query_word);
            query_phonemes       = get_phonemes(query_pronounciation);
            query_phonemes_count = query_phonemes.size();

#ifdef SIG_SLOT_AVAILABLE
            saw_strange_word(query_word, query_phonemes);
#endif
            for(const auto& [word, phonemes] : _phonemes_vector)
            {
                auto [dict_query_pronounciation,
                      dict_query_phonemes,
                      dict_query_phonemes_count] = phonemes;
                //TODO: delete find_and_replace logic and have const auto& after fixing \r endings of dictionary by getting a fresh dump
                akil::string::find_replace(dict_query_pronounciation, "\r", "");
                akil::string::find_replace(dict_query_pronounciation, "\n", "");
                if(akil::string::are_equal(dict_query_pronounciation, query_pronounciation))
                {
                    homonym_index_of_madeup_word = _phonemes_by_word.at(word);
                    break;
                }
            }
        }

        if(_mode == Use_More::And_More_Memory and not _is_levensthein_index
           and (synthetic ? homonym_index_of_madeup_word > -1 : true))
        {
            const auto query_word_indice =
                (synthetic ? homonym_index_of_madeup_word : _phonemes_by_word.at(query_word));
            const auto& distanced_items = _phoneme_index.at(query_word_indice);
            const ushort max_distance =
                instructions.max_distance > -1
                    ? std::min((size_t) instructions.max_distance, distanced_items.size())
                    : distanced_items.size();
            for(ushort distance = 0; distance <= max_distance; ++distance)
            {
                const auto& items_of_distance = distanced_items.at(distance);
                const auto result_range_begin = result.size();
                result.resize(result_range_begin + items_of_distance.size());
                std::transform(std::execution::unseq,
                               items_of_distance.cbegin(),
                               items_of_distance.cend(),
                               result.begin() + result_range_begin,
                               [&](const std::pair<size_t, char>& item) -> Alternative_Word
                               {
                                   return {
                                       _phonemes_vector.at(item.first).first,
                                       distance,
                                       (item.second ? (item.second > 0 ? "+" : "-") : "~")};
                               });
            }
            return result;
        }

        if(_mode == Use_More::And_More_Memory and not _is_levensthein_index
           and (synthetic and homonym_index_of_madeup_word == -1))
        {
            std::cout << "OOV procedure. Walking looong way now." << std::endl;
        }

        const ushort max_distance =
            instructions.max_distance > -1 ? instructions.max_distance : query_phonemes_count;
        std::mutex mutex;
        auto processor = [&](const std::string& dictionary_word,
                             const std::tuple<std::string, std::vector<std::string>, ushort>&
                                 dictionary_pronounciation_info)
        {
            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = dictionary_pronounciation_info;
            if(dictionary_word_phonemes_count == query_phonemes_count)
            {
                if(akil::string::are_equal(dictionary_word_pronounciation,
                                           query_pronounciation))
                {
                    if(not akil::string::are_equal(dictionary_word, query_word))
                    {
                        const std::lock_guard lock(mutex);
                        result.push_back({dictionary_word, 0, "~"});
                        return;
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
                        const std::lock_guard lock(mutex);
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
                            const std::lock_guard lock(mutex);
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
                        const std::lock_guard lock(mutex);
                        result.push_back({dictionary_word, distance, "+"});
                    }
                }
            }
        };

        std::for_each(std::execution::par_unseq,
                      _phonemes_vector.cbegin(),
                      _phonemes_vector.cend(),
                      [&](const auto& item)
                      {
                          const auto& [dictionary_word, dictionary_pronounciation_info] = item;

                          //            const bool has_parenthesis = akil::string::contains(dictionary_word, ')');
                          //            if(has_parenthesis)
                          //            {
                          //                continue;
                          //            }

                          processor(dictionary_word, dictionary_pronounciation_info);
                      });

#ifdef PROFILE_TIMING
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - chrono)
                                  .count();

        std::cout << "time_past_inside transcription_processor phonemes: " << duration
                  << std::endl;
#endif
        return result;
    }

    static inline std::vector<std::string> get_phonemes(const std::string& pronounciation)
    {
        return akil::string::split(pronounciation, ' ');
    }

  public:
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
                const auto& query_phoneme       = query_phonemes[query_phoneme_order];
                const auto& query_phoneme_size  = query_phoneme.size();
                const auto& query_phoneme_array = query_phoneme.c_str();

                for(ushort dictionary_phoneme_order = last_matched_dict_phoneme_order;
                    dictionary_phoneme_order < dictionary_phonemes_count;
                    ++dictionary_phoneme_order)
                {
                    const auto& dictionary_phoneme =
                        dictionary_phonemes[dictionary_phoneme_order];

                    if(dictionary_phoneme.size() == query_phoneme_size)
                    {
                        if(not memcmp(dictionary_phonemes[dictionary_phoneme_order].c_str(),
                                      query_phoneme_array,
                                      query_phoneme_size))
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
                const auto& query_phoneme_size  = query_phoneme.size();
                const auto& query_phoneme_array = query_phoneme.c_str();
                for(const auto& dictionary_phoneme : dictionary_phonemes)
                {
                    if(dictionary_phoneme.size() == query_phoneme_size)
                    {
                        if(not memcmp(dictionary_phoneme.c_str(),
                                      query_phoneme_array,
                                      query_phoneme_size))
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
    Use_More _mode = Use_More::CPU;
    Unordered_Map<std::string, size_t> _phonemes_by_word;
    std::vector<
        std::pair<std::string, std::tuple<std::string, std::vector<std::string>, ushort>>>
        _phonemes_vector;
    std::vector<std::vector<std::vector<std::pair<size_t, char>>>> _phoneme_index;
    bool _is_levensthein_index = false;
    Phoneme_Synthesizer _phoneme_synthesizer;
};

// --------------------------------------------------------------------------------------------------

std::vector<std::pair<std::string, std::string>>
Vocinity::Homophonic_Alternative_Composer::load_phonetics_dictionary(
    const std::filesystem::path& dictionary)
{
    std::vector<std::pair<std::string, std::string>> pronounciation_by_word;
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
                          << " is corrupted, 2 items expected in each line,\n entry of: |"
                          << line << "| is ignored" << std::endl;
            }
            if(akil::string::contains(word, "("))
            {
                word = akil::string::split(word, '(').at(0);
            }
            pronounciation_by_word.push_back({word, pronounciation});
        }
    }
    return pronounciation_by_word;
}

std::vector<std::vector<std::vector<std::pair<size_t, char>>>>
Vocinity::Homophonic_Alternative_Composer::
    precompute_phoneme_similarity_map_from_phonetics_dictionary(
        const std::filesystem::path& dictionary,
        const short max_distance,
        const ushort max_best_num_alternatives,
        const bool levenshtein)
{
    return precompute_phoneme_similarity_map_from_phonetics_dictionary(
        load_phonetics_dictionary(dictionary),
        max_distance,
        max_best_num_alternatives,
        levenshtein);
}

std::vector<std::vector<std::vector<std::pair<size_t, char>>>>
Vocinity::Homophonic_Alternative_Composer::
    precompute_phoneme_similarity_map_from_phonetics_dictionary(
        const std::vector<std::pair<std::string, std::string>>& dictionary,
        const short max_distance,
        const ushort max_best_num_alternatives,
        const bool levenshtein)
{
    std::vector<
        std::pair<std::string, std::tuple<std::string, std::vector<std::string>, ushort>>>
        phonemes_vector;

    std::vector<std::vector<std::vector<std::pair<size_t, char>>>> similarity_index;
    similarity_index.resize(dictionary.size());

#ifdef PROFILE_TIMING
    auto chrono = std::chrono::high_resolution_clock::now();
#endif
    for(auto [word, pronounciation] : dictionary)
    {
        akil::string::find_replace(pronounciation, "\r", "");
        akil::string::find_replace(pronounciation, "\n", "");
        const auto& phonemes     = akil::string::split(pronounciation, ' ');
        const auto phonemes_size = phonemes.size();
        const auto word_order    = phonemes_vector.size();
        const ushort max_distance_to_be_used =
            max_distance > -1 ? max_distance : phonemes_size;
        similarity_index[word_order].resize(max_distance_to_be_used + 1);
        phonemes_vector.push_back({word, {pronounciation, phonemes, phonemes_size}});
    }
#ifdef PROFILE_TIMING
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::high_resolution_clock::now() - chrono)
                              .count();
    std::cout << "parsing dictionary took: " << total_duration << std::endl;
#endif

    auto transcription_processor =
        [&](const size_t query_word_order,
            const std::string& query_word,
            const std::tuple<std::string, std::vector<std::string>, ushort>& outer_pron)
    {
        const auto& [query_pronounciation, query_phonemes, query_phonemes_count] = outer_pron;
        const ushort max_distance_to_be_used =
            max_distance > -1 ? max_distance : query_phonemes_count;

        for(size_t dictionary_order = 0; dictionary_order < phonemes_vector.size();
            ++dictionary_order)
        {
            const auto& [dictionary_word, inner_pron]    = phonemes_vector[dictionary_order];
            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = inner_pron;

            if(dictionary_word_phonemes_count == query_phonemes_count)
            {
                if(akil::string::are_equal(dictionary_word_pronounciation,
                                           query_pronounciation))
                {
                    if(not akil::string::are_equal(dictionary_word, query_word))
                    {
                        similarity_index[query_word_order][0].push_back({dictionary_order, 0});
                        continue;
                    }
                }
            }

            const ushort num_of_common_phonemes = Vocinity::Homophonic_Alternative_Composer::
                Homophonic_Alternative_Composer_Impl::get_num_of_common_phonemes(
                    query_phonemes,
                    query_phonemes_count,
                    dictionary_word_phonemes,
                    dictionary_word_phonemes_count,
                    false);

            for(ushort distance = 1; distance <= max_distance_to_be_used; ++distance)
            {
                if(num_of_common_phonemes == (query_phonemes_count - distance))
                {
                    if(dictionary_word_phonemes_count == query_phonemes_count - distance)
                    {
                        similarity_index[query_word_order][distance].push_back(
                            {dictionary_order, -1});
                    }
                    else if(dictionary_word_phonemes_count == query_phonemes_count)
                    {
                        const auto ordered_common_phonemes =
                            Vocinity::Homophonic_Alternative_Composer::
                                Homophonic_Alternative_Composer_Impl::
                                    get_num_of_common_phonemes(query_phonemes,
                                                               query_phonemes_count,
                                                               dictionary_word_phonemes,
                                                               dictionary_word_phonemes_count,
                                                               true);
                        if(ordered_common_phonemes == (query_phonemes_count - distance))
                        {
                            similarity_index[query_word_order]
                                .at(query_phonemes_count - ordered_common_phonemes)
                                .push_back({dictionary_order, 0});
                        }
                    }
                }
                else if(num_of_common_phonemes == query_phonemes_count)
                {
                    if(dictionary_word_phonemes_count == (query_phonemes_count + distance))
                    {
                        similarity_index[query_word_order][distance].push_back(
                            {dictionary_order, 1});
                    }
                }
            }
        }
    };

    auto levenshtein_processor =
        [&](const size_t query_word_order,
            const std::string& query_word,
            const std::tuple<std::string, std::vector<std::string>, ushort>& outer_pron)
    {
        const auto& [query_pronounciation, query_phonemes, query_phonemes_count] = outer_pron;

        for(size_t dictionary_order = 0; dictionary_order < phonemes_vector.size();
            ++dictionary_order)
        {
            const auto& [dictionary_word, inner_pron]    = phonemes_vector[dictionary_order];
            const auto& [dictionary_word_pronounciation,
                         dictionary_word_phonemes,
                         dictionary_word_phonemes_count] = inner_pron;

            if(dictionary_word_phonemes_count == query_phonemes_count)
            {
                if(akil::string::are_equal(dictionary_word_pronounciation,
                                           query_pronounciation))
                {
                    if(not akil::string::are_equal(dictionary_word, query_word))
                    {
                        similarity_index[query_word_order][0].push_back({dictionary_order, 0});
                        continue;
                    }
                }
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

                const char& final_op = additions > removals
                                           ? (additions > replacements ? 1 : 0)
                                           : (removals > replacements ? -1 : 0);
                similarity_index[query_word_order][levenshtein_difference.size()].push_back(
                    {dictionary_order, final_op});
            }
        }
    };


#ifdef PROFILE_TIMING
    chrono = std::chrono::high_resolution_clock::now();
#endif
    uint64_t progress_order = 0;
#ifdef __TBB_parallel_for_H
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, phonemes_vector.size()),
        [&](tbb::blocked_range<size_t> r)
        {
            for(size_t order = r.begin(); order < r.end(); ++order)
            {
                const auto& [query_word, outer_pron] = phonemes_vector.at(order);
                if(levenshtein)
                {
                    levenshtein_processor(order, query_word, outer_pron);
                    std::cout << ++progress_order << "/" << phonemes_vector.size()
                              << " indexed." << std::endl;
                }
                else
                {
                    transcription_processor(order, query_word, outer_pron);
                    std::cout << ++progress_order << "/" << phonemes_vector.size()
                              << " indexed." << std::endl;
                }

                if(max_best_num_alternatives)
                {
                    uint64_t num_items_we_have = 0;
                    for(ushort distance_order = 0; similarity_index[order].size();
                        ++distance_order)
                    {
                        auto& distanced_items = similarity_index[order][distance_order];
                        if(num_items_we_have > max_best_num_alternatives)
                        {
                            similarity_index[order].resize(distance_order);
                            break;
                        }
                        num_items_we_have += distanced_items.size();
                        if(num_items_we_have > max_best_num_alternatives)
                        {
                            std::sort(std::execution::unseq,
                                      distanced_items.begin(),
                                      distanced_items.end(),
                                      [](const std::pair<size_t, char>& one,
                                         const std::pair<size_t, char>& another) -> bool
                                      {
                                          const auto& [_, first_op]   = one;
                                          const auto& [__, second_op] = another;

                                          if(first_op == 0 and (second_op not_eq 0))
                                          {
                                              return true;
                                          }
                                          else if((first_op not_eq 0) and (second_op not_eq 0))
                                          {
                                              if(first_op == -1 and second_op == 1)
                                              {
                                                  return true;
                                              }
                                          }
                                          return false;
                                      });
                            distanced_items.resize(max_best_num_alternatives
                                                   - num_items_we_have);
                        }
                    }
                }
            }
        });
#else
    for(size_t order = 0; order < phonemes_vector.size(); ++order)
    {
        const auto& [query_word, outer_pron] = phonemes_vector.at(order);
        if(levenshtein)
        {
            levenshtein_processor(order, query_word, outer_pron);
            std::cout << ++progress_order << "/" << phonemes_vector.size() << " indexed."
                      << std::endl;
        }
        else
        {
            transcription_processor(order, query_word, outer_pron);
            std::cout << ++progress_order << "/" << phonemes_vector.size() << " indexed."
                      << std::endl;
        }

        if(max_best_num_alternatives)
        {
            uint64_t num_items_we_have = 0;
            for(ushort distance_order = 0; similarity_index[order].size(); ++distance_order)
            {
                auto& distanced_items = similarity_index[order][distance_order];
                if(num_items_we_have > max_best_num_alternatives)
                {
                    similarity_index[order].resize(distance_order);
                    break;
                }
                num_items_we_have += distanced_items.size();
                if(num_items_we_have > max_best_num_alternatives)
                {
                    std::sort(std::execution::unseq,
                              distanced_items.begin(),
                              distanced_items.end(),
                              [](const std::pair<size_t, char>& one,
                                 const std::pair<size_t, char>& another) -> bool
                              {
                                  const auto& [_, first_op]   = one;
                                  const auto& [__, second_op] = another;

                                  if(first_op == 0 and (second_op not_eq 0))
                                  {
                                      return true;
                                  }
                                  else if((first_op not_eq 0) and (second_op not_eq 0))
                                  {
                                      if(first_op == -1 and second_op == 1)
                                      {
                                          return true;
                                      }
                                  }
                                  return false;
                              });
                    distanced_items.resize(max_best_num_alternatives - num_items_we_have);
                }
            }
        }
    }
#endif
#ifdef PROFILE_TIMING
    total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - chrono)
                         .count();
    std::cout << "pre-computation took: " << total_duration << std::endl;
#endif

    return similarity_index;
}

void
Vocinity::Homophonic_Alternative_Composer::save_precomputed_phoneme_similarity_map(
    const std::vector<std::vector<std::vector<std::pair<size_t, char>>>>& map,
    const std::filesystem::path& map_path_to_be_exported,
    const bool binary)
{
    const nlohmann::json j_map(map);
    if(binary)
    {
        const auto& cbor = nlohmann::json::to_cbor(j_map);
        std::ofstream file_handle(map_path_to_be_exported, std::ios::binary | std::ios::out);
        std::copy(cbor.cbegin(), cbor.cend(), std::ostreambuf_iterator<char>(file_handle));
    }
    else
    {
        std::ofstream file_handle(map_path_to_be_exported);
        file_handle << j_map;
    }
}

std::vector<std::vector<std::vector<std::pair<size_t, char>>>>
Vocinity::Homophonic_Alternative_Composer::load_precomputed_phoneme_similarity_map(
    const std::filesystem::path& map_path_to_be_imported,
    const bool binary)
{
    nlohmann::json similarity_map_json;
    if(binary)
    {
        std::ifstream file_handle(map_path_to_be_imported, std::ios::binary | std::ios::in);
        assert(file_handle.good() && "file not exists");
        const auto file_size = std::filesystem::file_size(map_path_to_be_imported);
        if(file_size == 0)
        {
            std::cout << "File is empty: " << map_path_to_be_imported << std::endl;
            return {};
        }
        std::vector<std::uint8_t> buffer(file_size);
        file_handle.read(reinterpret_cast<char*>(buffer.data()), file_size);
        similarity_map_json = nlohmann::json::from_cbor(buffer);
    }
    else
    {
        std::ifstream file_handle(map_path_to_be_imported, std::ios::in);
        assert(file_handle.good() && "file not exists");
        file_handle >> similarity_map_json;
    }

    return similarity_map_json
        .get<std::vector<std::vector<std::vector<std::pair<size_t, char>>>>>();
}

// --------------------------------------------------------------------------------------------------

Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer(
    const std::vector<std::pair<std::string, std::string>>& phonetics_dictionary)
    : _impl(std::make_unique<
            Vocinity::Homophonic_Alternative_Composer::Homophonic_Alternative_Composer_Impl>(
        phonetics_dictionary))
{
#ifdef SIG_SLOT_AVAILABLE
    _impl->saw_strange_word.connect(
        [this](const std::string word, const std::vector<std::string> phonemes)
        { saw_strange_word(std::move(word), std::move(phonemes)); });
#endif
}

Vocinity::Homophonic_Alternative_Composer::~Homophonic_Alternative_Composer()
{}

void
Vocinity::Homophonic_Alternative_Composer::set_precomputed_phoneme_similarity_map(
    std::vector<std::vector<std::vector<std::pair<size_t, char>>>>&& map,
    const bool levenshtein)
{
    if(map.empty())
    {
        return;
    }
    _impl->set_precomputed_phoneme_similarity_map(map, levenshtein);
}

Vocinity::Homophonic_Alternative_Composer::Alternative_Words_Of_Sentence
Vocinity::Homophonic_Alternative_Composer::get_alternatives(const std::string& reference,
                                                            const Instructions& instructions,
                                                            const bool parallel)
{
    //  const std::lock_guard lock(_mutex);
#ifdef PROFILE_TIMING
    auto chrono = std::chrono::high_resolution_clock::now();
#endif
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

    sentence_result.resize(words.size());
#ifdef __TBB_parallel_for_H
    if(parallel)
    {
        tbb::parallel_for(
            tbb::blocked_range<int>(0, words.size()),
            [&](tbb::blocked_range<int> r)
            {
                for(uint i = r.begin(); i < r.end(); ++i)
                {
                    const auto& word = words.at(i);
                    if(akil::memory::vector_contains(instructions.dismissed_word_indices, i))
                    {
                        sentence_result[i] = {{}};
                        return;
                    }

                    if(akil::memory::vector_contains(instructions.dismissed_words, word))
                    {
                        sentence_result[i] = {{}};
                        return;
                    }

                    sentence_result[i] = _impl->get_alternatives(word, instructions, parallel);
                }
            });
    }
    else
#endif
    {
        for(uint order = 0; order < words.size(); ++order)
        {
            const auto& word = words.at(order);
            if(akil::memory::vector_contains(instructions.dismissed_word_indices, order))
            {
                sentence_result[order] = {{}};
                continue;
            }

            if(akil::memory::vector_contains(instructions.dismissed_words, word))
            {
                sentence_result[order] = {{}};
                continue;
            }

            sentence_result[order] = _impl->get_alternatives(word, instructions, parallel);
        }
    }

#ifdef PROFILE_TIMING
    const auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::high_resolution_clock::now() - chrono)
                                    .count();
    std::cout << "time spent in total: " << total_duration << std::endl;
#endif
    return sentence_result;
}
