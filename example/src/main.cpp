#include "../src/Context_Scorer.hpp"

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
            Vocinity::Context_Scorer::Model_Family::OpenAI,
            Vocinity::Context_Scorer::Tokenizer_Configuration{argv[2], argv[3]},
            Vocinity::Context_Scorer::Inference_Backend::CUDA};

        std::cout << "Instance " << instance_index << " " << argv[1]
                  << " model initialization took: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now()
                         - model_initialization_chrono)
                         .count()
                  << " milliseconds\n\n";

        std::vector<std::pair<std::string, Vocinity::Context_Scorer::Score>> scores;
        for(uint64_t utterance_order = 0; utterance_order < utterances.size();
            ++utterance_order)
        {
            if(utterance_order % instance_index)
            {
                continue;
            }
            const auto& utterance = utterances.at(utterance_order);
            auto inference_chrone = std::chrono::high_resolution_clock::now();
            const auto& score     = scorer.score(utterance, true);
            scores.push_back({utterance, score});
            const auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - inference_chrone)
                    .count();

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

        return scores;
    };

    Vocinity::Homophonic_Alternative_Composer composer;
    Vocinity::Homophonic_Alternative_Composer::Instructions instructions;
    instructions.max_distance              = 2;
    instructions.max_best_num_alternatives = 10;
    //instructions.dismissed_word_indices    = {0, 1, 2, 3};
    instructions.method =
        Vocinity::Homophonic_Alternative_Composer::Matching_Method::Phoneme_Transcription;

    const auto input                   = std::string(argv[5]);
    const auto& splitted_raw_sentences = akil::string::split(input, '.');
    std::cout << "----------------------------------------------------------------------------"
                 "-------------------------"
              << std::endl;
    std::cout <<"Input is: \""<< input <<"\""<< std::endl;
    std::cout << "----------------------------------------------------------------------------"
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
    std::cout << "----------------------------------------------------------------------------"
                 "-------------------------"
              << std::endl;

    std::vector<std::vector<std::vector<std::string>>> combinations;
    combinations.resize(splitted_raw_sentences.size());
    ushort sentence_order = 0;
    for(const auto& sentence : splitted_raw_sentences)
    {
        std::vector<std::string> raw_words = akil::string::split(sentence, ' ');
        if(not instructions.dismissed_words.empty())
        {
            raw_words = akil::string::split(sentence, ' ');
        }

        auto word_combinations    = composer.get_alternatives(sentence, instructions);
        const double warmup_count = 100;
        auto inference_chrone     = std::chrono::high_resolution_clock::now();
        for(int warmup = 0; warmup < warmup_count; ++warmup)
        {
            word_combinations = composer.get_alternatives(sentence, instructions, false);
        }
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - inference_chrone)
                                  .count();
        std::cout << "Homonym generation took " << duration / warmup_count << " msecs"
                  << std::endl;
        std::cout << "------------------------------------------------------------------------"
                     "-----------------------------"
                  << std::endl;
        combinations[sentence_order].push_back(akil::string::split(sentence, ' '));
        for(ushort block_order = 0; block_order < word_combinations.size(); ++block_order)
        {
            {
                const auto& word_alternatives = word_combinations.at(block_order);
                for(const auto& alternative : word_alternatives)
                {
                    const auto& [similar_word, distance, op] = alternative;
                    std::cout << std::string(block_order, '\t') << similar_word << " (" << op
                              << distance << ")" << std::endl;
                }
            }
            if(not instructions.dismissed_word_indices.empty())
            {
                if(akil::memory::vector_contains(instructions.dismissed_word_indices,
                                                 (ushort) block_order))
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

            const auto past = combinations.at(sentence_order);
            for(auto past_sentence : past)
            {
                const auto& word_alternatives = word_combinations.at(block_order);
                for(const auto& alternative : word_alternatives)
                {
                    const auto& [similar_word, distance, op] = alternative;

                    past_sentence[block_order] = similar_word;
                    combinations[sentence_order].push_back(past_sentence);
                }
            }
        }
        ++sentence_order;

        std::cout << "------------------------------------------------------------------------"
                     "-----------------------------"
                  << std::endl;
    }

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
            auto full_statement=context_helper + " " + alternative;
            std::transform(
                full_statement.cbegin(),
                full_statement.cend(),
                full_statement.begin(),
                [](const auto& c)
                { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
            utterances.push_back(full_statement);

            std::cout << alternative << std::endl;
        }

        std::cout << "------------------------------------------------------------------------"
                     "-----------------------------"
                  << std::endl;
    }

    std::vector<std::pair<std::string, Vocinity::Context_Scorer::Score>> context_scores;
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

        std::vector<
            std::future<std::vector<std::pair<std::string, Vocinity::Context_Scorer::Score>>>>
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

    const bool parallel = true;
    if(parallel)
    {
#ifdef CPP17_AVAILABLE
        std::sort(std::execution::par_unseq,
                  context_scores.begin(),
                  context_scores.end(),
                  [](const auto& one, const auto& another) -> bool
                  { return one.second.mean > another.second.mean; });
#else
        __gnu_parallel::sort(context_scores.begin(),
                             context_scores.end(),
                             [](const auto& one, const auto& another) -> bool
                             { return one.second.mean > another.second.mean; });
#endif
    }
    else
    {
        std::sort(context_scores.begin(),
                  context_scores.end(),
                  [](const auto& one, const auto& another) -> bool
                  { return one.second.mean > another.second.mean; });
    }

    uint64_t entry_order = 0;
    for(const auto& entry : context_scores)
    {
        if(entry_order++ < 5)
        {
            const auto& score = entry.second;
            std::cout << "Sentence: " << entry.first << std::endl;
            std::cout << "negative_log_likelihood: " << score.negative_log_likelihood
                      << std::endl;
            std::cout << "production: " << score.production << std::endl;
            std::cout << "mean: " << score.mean << std::endl;
            std::cout << "g_mean: " << score.g_mean << std::endl;
            std::cout << "h_mean: " << score.h_mean << std::endl;
            std::cout << "loss: " << score.loss << std::endl;
            std::cout << "sentence_probability: " << score.sentence_probability << std::endl;
            std::cout << "----------------------------------------------------------------------------"
                         "-------------------------"
                      << std::endl;
        }
        else
        {
            std::cout <<"mean perplexity: " <<entry.second.mean << " | " << entry.first << std::endl<< std::endl;
        }
    }

    return 0;
}
