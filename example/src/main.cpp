#include "../src/Context-Scorer.hpp"
#include "../src/Homophonic-Alternatives.hpp"

/**
 * @brief main
 * @param argc program + 5 arguments
 * @param argv [1] is the model, argv[2] is vocab.json, argv[3] is merges.txt, argv[4] is device as either --cpu or --cuda,
 *  argv[5] is the sentence in any length
 * /path/to/Context-Scorer ./model.pt ./vocab.json ./merges.txt --cuda "I like it."
 * @return
 */
int
main(int argc, char* argv[])
{
    setlocale(LC_NUMERIC, "C");
    const auto physical_cores = std::thread::hardware_concurrency() / 2;
    std::cout << physical_cores << " physical cores available." << std::endl;
    std::cout << argv[4] << " device is selected" << std::endl;

    auto inference = [&](const int instance_index,
                         const std::vector<std::string>& utterances,
                         const bool verbose = false)
    {
        auto model_initialization_chrono = std::chrono::high_resolution_clock::now();
        Vocinity::Context_Scorer scorer{
            argv[1],
            Vocinity::Context_Scorer::GPT_TYPE::DistilGPT2,
            Vocinity::Context_Scorer::Tokenizer_Configuration{argv[2], argv[3]},
            Vocinity::Context_Scorer::Precision::FP32,
            std::string(argv[4]) == "--cuda"
                ? Vocinity::Context_Scorer::Inference_Hardware::CUDA
                : Vocinity::Context_Scorer::Inference_Hardware::CPU};
            scorer.flush_cuda_tensor_cache_before_inference();

        std::cout << "Instance " << instance_index << " " << argv[1]
                  << " model initialization took: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now()
                         - model_initialization_chrono)
                         .count()
                  << " milliseconds\n\n";

        bool batching = true;
        std::vector<std::tuple<std::string, Vocinity::Context_Scorer::Score, long>> scores;
        if(batching)
        {
            scorer.score_contexts(utterances, true);

            auto inference_chrono = std::chrono::high_resolution_clock::now();
            const auto& results   = scorer.score_contexts(utterances, true);
            const auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - inference_chrono)
                    .count();

            for(uint64_t utterance_order = 0; utterance_order < utterances.size();
                ++utterance_order)
            {
                if(utterance_order % instance_index)
                {
                    continue;
                }

                const auto& utterance = utterances.at(utterance_order);
                const auto& score     = results.at(utterance_order);
                scores.push_back({utterance, score, duration});
                if(verbose)
                {
                    std::cout << instance_index << " Inference took " << duration
                              << " milliseconds." << std::endl;
                    std::cout << "Instance " << instance_index << " sentence: " << utterance
                              << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "negative_log_likelihood: " << score.negative_log_likelihood
                              << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "production: " << score.production << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "mean: " << score.mean << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "g_mean: " << score.g_mean << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "h_mean: " << score.h_mean << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "loss: " << score.loss << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "sentence_probability: " << score.sentence_probability
                              << std::endl;
                }
            }
        }
        else
        {
            for(uint64_t utterance_order = 0; utterance_order < utterances.size();
                ++utterance_order)
            {
                if(utterance_order % instance_index)
                {
                    continue;
                }

                const auto& utterance = utterances.at(utterance_order);
                scorer.score_context(utterance, false);

                auto inference_chrono = std::chrono::high_resolution_clock::now();
                const auto& score     = scorer.score_context(utterance, true);
                const auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - inference_chrono)
                        .count();
                scores.push_back({utterance, score, duration});

                if(verbose)
                {
                    std::cout << instance_index << " Inference took " << duration
                              << " milliseconds." << std::endl;
                    std::cout << "Instance " << instance_index << " sentence: " << utterance
                              << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "negative_log_likelihood: " << score.negative_log_likelihood
                              << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "production: " << score.production << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "mean: " << score.mean << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "g_mean: " << score.g_mean << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "h_mean: " << score.h_mean << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "loss: " << score.loss << std::endl;
                    std::cout << "Instance " << instance_index << " "
                              << "sentence_probability: " << score.sentence_probability
                              << std::endl;
                }
            }
        }

        return scores;
    };


    std::vector<std::vector<std::vector<std::string>>> combinations;
    if(true)
    {
        const auto& phonetics_dictionary =
            Vocinity::Homophonic_Alternative_Composer::load_phonetics_dictionary(
                "/opt/cloud/projects/vocinity/models/context-scorer/cmudict-0.7b.txt");

        Vocinity::Homophonic_Alternative_Composer composer{phonetics_dictionary};
        Vocinity::Homophonic_Alternative_Composer::Instructions instructions;
        instructions.max_distance              = 2;
        instructions.max_best_num_alternatives = 3;
        //  instructions.dismissed_word_indices    = {0, 1, 2,4,5,6};
        instructions.method =
            Vocinity::Homophonic_Alternative_Composer::Matching_Method::Phoneme_Transcription;

        if(false)
        {
            if(instructions.method
               == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
                   Phoneme_Transcription)
            {
                const auto similarity_map_composed =
                    Vocinity::Homophonic_Alternative_Composer::
                        precompute_phoneme_similarity_map_from_phonetics_dictionary(
                            phonetics_dictionary, 2, 0, false);
                Vocinity::Homophonic_Alternative_Composer::
                    save_precomputed_phoneme_similarity_map(
                        similarity_map_composed,
                        "./similarity_map-cmudict07b-dist2-phoneme_transcription.cbor",
                        true);
            }
            else if(instructions.method
                    == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
                        Phoneme_Levenshtein)
            {
                const auto similarity_map_composed =
                    Vocinity::Homophonic_Alternative_Composer::
                        precompute_phoneme_similarity_map_from_phonetics_dictionary(
                            phonetics_dictionary, 2, 0, true);
                Vocinity::Homophonic_Alternative_Composer::
                    save_precomputed_phoneme_similarity_map(
                        similarity_map_composed,
                        "./similarity_map-cmudict07b-dist2-phoneme_levenshtein.cbor",
                        true);
            }

            return 0;
        }

        if(true)
        {
            if(instructions.method
               == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
                   Phoneme_Transcription)
            {
                auto similarity_map = Vocinity::Homophonic_Alternative_Composer::
                    load_precomputed_phoneme_similarity_map(
                        "/opt/cloud/projects/vocinity/models/context-scorer/"
                        "similarity_map-cmudict07b-dist2-phoneme_transcription.cbor",
                        true);

                composer.set_precomputed_phoneme_similarity_map(std::move(similarity_map),
                                                                false);
            }
            else if(instructions.method
                    == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
                        Phoneme_Levenshtein)
            {
                auto similarity_map = Vocinity::Homophonic_Alternative_Composer::
                    load_precomputed_phoneme_similarity_map(
                        "/opt/cloud/projects/vocinity/models/context-scorer/"
                        "similarity_map-cmudict07b-dist2-phoneme_levenshtein.cbor",
                        true);

                composer.set_precomputed_phoneme_similarity_map(std::move(similarity_map),
                                                                true);
            }
        }

        const auto input                   = std::string(argv[5]);
        const auto& splitted_raw_sentences = akil::string::split(input, '.');
        std::cout
            << "----------------------------------------------------------------------------"
               "-------------------------"
            << std::endl;
        std::cout << "Input is: \"" << input << "\"" << std::endl;
        std::cout
            << "----------------------------------------------------------------------------"
               "-------------------------"
            << std::endl;
        std::cout << instructions.max_best_num_alternatives
                  << " best alternative(s) wanted and max allowed variational distance is "
                  << instructions.max_distance << ". "
                  << (not(instructions.dismissed_words.empty()
                          and instructions.dismissed_word_indices.empty())
                          ? "Words "
                          : "No words dismissed.");
        if(not(instructions.dismissed_words.empty()
               and instructions.dismissed_word_indices.empty()))
        {
            std::string dismissed_indices;
            dismissed_indices += "{ ";
            for(auto item : instructions.dismissed_word_indices)
            {
                dismissed_indices += std::to_string(item) + ", ";
            }
            dismissed_indices.resize(dismissed_indices.size() - 2);
            dismissed_indices += "} ";
            std::cout << dismissed_indices;
            for(auto item : instructions.dismissed_words)
            {
                std::cout << item << ' ';
            }
            std::cout << "are dismissed.";
        }
        std::cout << std::endl;
        std::cout
            << "----------------------------------------------------------------------------"
               "-------------------------"
            << std::endl;

        combinations.resize(splitted_raw_sentences.size());
        ushort sentence_order = 0;
        for(const auto& sentence : splitted_raw_sentences)
        {
            std::vector<std::string> raw_words = akil::string::split(sentence, ' ');
            if(not instructions.dismissed_words.empty())
            {
                raw_words = akil::string::split(sentence, ' ');
            }
            auto warmup_combinations = composer.get_alternatives(sentence, instructions);
            auto chrono              = std::chrono::high_resolution_clock::now();
            const auto word_combinations =
                composer.get_alternatives(sentence, instructions, true);
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::high_resolution_clock::now() - chrono)
                                      .count();
            std::cout << "Homonym generation took " << duration << " msecs" << std::endl;
            std::cout
                << "------------------------------------------------------------------------"
                   "-----------------------------"
                << std::endl;

            combinations[sentence_order].push_back(akil::string::split(sentence, ' '));
            for(uint block_order = 0; block_order < word_combinations.size(); ++block_order)
            {
                {
                    const auto& word_alternatives = word_combinations.at(block_order);
                    for(const auto& alternative : word_alternatives)
                    {
                        const auto& [similar_word, distance, op] = alternative;
                        std::cout << std::string(block_order, '\t') << similar_word << " ("
                                  << op << distance << ")" << std::endl;
                    }
                }
                if(not instructions.dismissed_word_indices.empty())
                {
                    if(akil::memory::vector_contains(instructions.dismissed_word_indices,
                                                     block_order))
                    {
                        continue;
                    }
                }

                if(not instructions.dismissed_words.empty())
                {
                    if(akil::memory::vector_contains(instructions.dismissed_words,
                                                     raw_words.at(block_order)))
                    {
                        continue;
                    }
                }

                const size_t past_items_count = combinations.at(sentence_order).size();
                for(uint64_t past_item_order = 0; past_item_order < past_items_count;
                    ++past_item_order)
                {
                    const auto& word_alternatives = word_combinations.at(block_order);
                    for(const auto& alternative : word_alternatives)
                    {
                        const auto& [similar_word, distance, op] = alternative;
                        auto past_sentence = combinations[sentence_order][past_item_order];
                        past_sentence[block_order] = similar_word;
                        combinations[sentence_order].push_back(past_sentence);
                    }
                }
            }
            ++sentence_order;

            std::cout
                << "------------------------------------------------------------------------"
                   "-----------------------------"
                << std::endl;
        }
    }
    //	combinations.resize(1);
    //	combinations[0].push_back({"smart", "rower"});
    //	combinations[0].push_back({"smart", "roher"});
    const std::string context_helper =
        "Click on the eye in the icon tray to pick your product of interest or say "
        "echelon-connect bike or smart rower.";

    std::vector<std::string> utterances;
    for(const auto& sentence : combinations)
    {
        for(const auto& words : sentence)
        {
            std::string alternative;
            for(const auto& word : words)
            {
                alternative += word + " ";
            }

            alternative.resize(alternative.size() - 1);
            alternative += ".";
            auto full_statement = context_helper + " " + alternative;

#ifdef CPP17_AVAILABLE
            std::transform(
                std::execution::unseq,
                full_statement.cbegin(),
                full_statement.cend(),
                full_statement.begin(),
                [](const auto& c)
                { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
#else
            std::transform(
                full_statement.cbegin(),
                full_statement.cend(),
                full_statement.begin(),
                [](const auto& c)
                { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
#endif

            utterances.push_back(full_statement);

            std::cout << alternative << std::endl;
        }

        std::cout << "------------------------------------------------------------------------"
                     "-----------------------------"
                  << std::endl;
    }

    std::vector<std::tuple<std::string, Vocinity::Context_Scorer::Score, long>> context_scores;
    // as you see these are static functions and affect all instances
    // NOTE THAT IF YOUR TORCH CONFIGURATION USES OMP UNDER THE HOOD INSTEAD OF MKL's
    // then you are setting up entire omp thread pool for your remaining part of the
    // program.
    const bool only_instance = true;
    if(only_instance)
    {
        /// well it is almost always better to use single interop and intraop thread for this model
        Vocinity::Context_Scorer::
            optimize_parallelization_policy_for_use_of_multiple_instances();

        const auto& instance_score = inference(1, utterances);
        context_scores.insert(
            std::end(context_scores), std::begin(instance_score), std::end(instance_score));
    }
    else
    {
        Vocinity::Context_Scorer::
            optimize_parallelization_policy_for_use_of_multiple_instances();

        std::vector<std::future<
            std::vector<std::tuple<std::string, Vocinity::Context_Scorer::Score, long>>>>
            instances;
        for(int instance_index = 0; instance_index < physical_cores; ++instance_index)
        {
            instances.emplace_back(
                std::async(std::launch::async, inference, instance_index, utterances));
        }
        for(int instance_index = 0; instance_index < physical_cores; ++instance_index)
        {
            const auto& instance_score = instances.at(instance_index).get();
            context_scores.insert(std::end(context_scores),
                                  std::begin(instance_score),
                                  std::end(instance_score));
        }
    }


#ifdef CPP17_AVAILABLE
    std::sort(std::execution::unseq,
              context_scores.begin(),
              context_scores.end(),
              [](const auto& one, const auto& another) -> bool
              {
                  const auto& [_, first_score, __]       = one;
                  const auto& [___, second_score, _____] = another;
                  return first_score.mean > second_score.mean;
              });
#else
    std::sort(context_scores.begin(),
              context_scores.end(),
              [](const auto& one, const auto& another) -> bool
              {
                  const auto& [_, first_score, __] = one;
                  const auto& [___, second_score, _____] = another;
                  return first_score.mean > second_score.mean;
              });
#endif

    uint64_t entry_order = 0;
    for(const auto& [utterence, score, time] : context_scores)
    {
        if(entry_order++ < 5)
        {
            std::cout << "Sentence: " << utterence << std::endl;
            std::cout << "Duration: " << time << std::endl;
            std::cout << "negative_log_likelihood: " << score.negative_log_likelihood
                      << std::endl;
            std::cout << "production: " << score.production << std::endl;
            std::cout << "mean: " << score.mean << std::endl;
            std::cout << "g_mean: " << score.g_mean << std::endl;
            std::cout << "h_mean: " << score.h_mean << std::endl;
            std::cout << "loss: " << score.loss << std::endl;
            std::cout << "sentence_probability: " << score.sentence_probability << std::endl;
            std::cout << "--------------------------------------------------------------------"
                         "--------"
                         "-------------------------"
                      << std::endl;
        }
        else
        {
            std::cout << "mean perplexity: " << score.mean << " | " << utterence
                      << " | Duration: " << time << std::endl
                      << std::endl
                      << std::endl;
        }
    }

    return 0;
}
