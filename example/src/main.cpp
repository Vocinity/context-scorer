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

    auto inference = [=](const int instance_index)
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

        auto inference_chrone = std::chrono::high_resolution_clock::now();
        const auto score      = scorer.score(argv[5], true);
        const auto duration   = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - inference_chrone)
                                  .count();
        std::cout << instance_index << " Inference took " << duration << " milliseconds." << std::endl;

        std::cout << "Instance " << instance_index << " sentence: "
                  << argv[5] << std::endl;
        std::cout << "Instance " << instance_index << " "
                  << "negative_log_likelihood: " << score.negative_log_likelihood << std::endl;
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
                  << "sentence_probability: " << score.sentence_probability << std::endl;
    };

    /// as you see these are static functions and affect all instances
    /// NOTE THAT IF YOUR TORCH CONFIGURATION USES OMP UNDER THE HOOD INSTEAD OF MKL's
    /// then you are setting up entire omp thread pool for your remaining part of the
    /// program.
    ///
    const bool only_instance = true;
    if(only_instance)
    {
        /// well it is almost always better to use single interop and intraop thread for this model
                Vocinity::Context_Scorer::
                    optimize_parallelization_policy_for_use_of_multiple_instances();

        inference(1);
    }
    else
    {
        Vocinity::Context_Scorer::
            optimize_parallelization_policy_for_use_of_multiple_instances();

        std::vector<std::future<void>> instances;
        for(int instance_index = 0; instance_index < physical_cores; ++instance_index)
        {
            instances.emplace_back(std::async(std::launch::async, inference, instance_index));
        }
        for(int instance_index = 0; instance_index < physical_cores; ++instance_index)
        {
            instances.at(instance_index).get();
        }
    }

    return 0;
}
