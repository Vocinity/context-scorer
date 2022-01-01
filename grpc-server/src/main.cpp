#include "../src/Context-Scorer.hpp"
#include "../src/Homophonic-Alternatives.hpp"
#include "context-scorer.grpc.pb.h"
#include "context-scorer.pb.h"

#include <grpc++/grpc++.h>

class Context_Scorer_Server
{
  public:
    using Model_Code = std::string;

    struct Homonym_Composer_Configuration
    {
        Homonym_Composer_Configuration()
        {
            id = akil::functional::getDefinitelyUniqueId<uint32_t>("homonym_composer_id");
        }
        std::filesystem::path dictionary_path;
        std::filesystem::path precomputed_phoneme_similarity_map;
        ushort max_distance = 0;
        Vocinity::Homophonic_Alternative_Composer::Matching_Method matching_method;
        uint32_t id              = 0;
        bool is_levenshtein_dump = false;
    };
    using Levenshtein_Homonym_Composer_Configuration   = Homonym_Composer_Configuration;
    using Phoneme_Based_Homonym_Composer_Configuration = Homonym_Composer_Configuration;
    struct Scorer_Model_Configuration
    {
        uint32_t homonym_composer_configuration_id = 0;
        std::filesystem::path model_path;
        Vocinity::Context_Scorer::GPT_TYPE type =
            Vocinity::Context_Scorer::GPT_TYPE::DistilGPT2;
        Vocinity::Context_Scorer::Tokenizer_Configuration tokenizer_configuration;
        Vocinity::Context_Scorer::Precision precision =
            Vocinity::Context_Scorer::Precision::FP32;
        Vocinity::Context_Scorer::Inference_Hardware hardware =
            Vocinity::Context_Scorer::Inference_Hardware::CPU;
    };

  public:
    class Service final : public Vocinity::Context_Scorer_gRPC::Service
    {
    public:
        void initialize(
            Unordered_Map<Model_Code,
                          std::pair<Scorer_Model_Configuration,
                                    std::pair<Phoneme_Based_Homonym_Composer_Configuration,

                                              Levenshtein_Homonym_Composer_Configuration>>>&&
                configuration)
        {
            _model_configurations = std::move(configuration);
        }

      public:
        virtual grpc::Status say_hi(grpc::ServerContext* context,
                                    const Vocinity::Knock_Knock* request,
                                    Vocinity::Session_Ticket* reply) override
        {
            const uint32_t client_id =
                akil::functional::getDefinitelyUniqueId<uint32_t>("grpc_client_id");
            const auto wanted_models_size = request->models_that_you_planning_to_use_size();
            for(int model_order = 0; model_order < wanted_models_size; ++model_order)
            {
                const auto wanted_model =
                    request->models_that_you_planning_to_use(model_order);
                const std::string model_code =
                    wanted_model.code().empty() ? "generic" : wanted_model.code();

                if(not _model_configurations.contains(model_code))
                {
                    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                                        "There is no such model as: " + model_code);
                }

                create_processors(model_code);

                if(not akil::memory::vector_contains(_clients.at(model_code), client_id))
                {
                    _clients[model_code].push_back(client_id);
                }
            }

            reply->set_id(client_id);

            return grpc::Status::OK;
        }

        virtual grpc::Status get_homonyms(grpc::ServerContext* context,
                                          const Vocinity::Homonym_Generation_Query* request,
                                          Vocinity::Homonyms* reply) override
        {
            const auto& model_code = request->model_code();

            if(not _model_configurations.contains(model_code))
            {
                return grpc::Status(grpc::StatusCode::NOT_FOUND,
                                    "There is no such model as: " + model_code);
            }

            const auto& input = request->input();

            const auto& instructions = Vocinity::Homophonic_Alternative_Composer::Instructions{
                (ushort) request->max_num_of_best_num_homonyms(),
                (short) request->max_distance(),
                get_vector_from_repeated(request->dismissed_word_indices()),
                get_vector_from_repeated(request->dismissed_words()),
                static_cast<Vocinity::Homophonic_Alternative_Composer::Matching_Method>(
                    request->matching_method())};

            const bool is_levenshtein =
                static_cast<Vocinity::Homophonic_Alternative_Composer::Matching_Method>(
                    request->matching_method())
                == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
                    Phoneme_Levenshtein;

            const auto& homonym_model_configuration =
                is_levenshtein ? _model_configurations.at(model_code).second.second
                               : _model_configurations.at(model_code).second.first;

            auto composer = is_levenshtein
                                ? _homonym_composers.at(homonym_model_configuration.id).second
                                : _homonym_composers.at(homonym_model_configuration.id).first;

            const auto& combinations = run_homonoym_composing(input, composer, instructions);

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
                    auto& full_statement = alternative;

#ifdef CPP17_AVAILABLE
                    std::transform(std::execution::unseq,
                                   full_statement.cbegin(),
                                   full_statement.cend(),
                                   full_statement.begin(),
                                   [](const auto& c) {
                                       return static_cast<char>(
                                           std::tolower(static_cast<unsigned char>(c)));
                                   });
#else
                    std::transform(full_statement.cbegin(),
                                   full_statement.cend(),
                                   full_statement.begin(),
                                   [](const auto& c) {
                                       return static_cast<char>(
                                           std::tolower(static_cast<unsigned char>(c)));
                                   });
#endif

                    reply->add_alternatives_of_input(full_statement);
                }
            }

            return grpc::Status::OK;
        }

        virtual grpc::Status get_best_n_alternatives(
            grpc::ServerContext* context,
            const Vocinity::Context_Scoring_Query* request,
            Vocinity::Context_Score* reply) override
        {
            const auto& material             = request->material();
            const auto homonym_query_request = material.input();
            const auto& model_code           = homonym_query_request.model_code();

            if(not _model_configurations.contains(model_code))
            {
                return grpc::Status(grpc::StatusCode::NOT_FOUND,
                                    "There is no such model as: " + model_code);
            }

            const auto& input = homonym_query_request.input();

            const auto& instructions = Vocinity::Homophonic_Alternative_Composer::Instructions{
                (ushort) homonym_query_request.max_num_of_best_num_homonyms(),
                (short) homonym_query_request.max_distance(),
                get_vector_from_repeated(homonym_query_request.dismissed_word_indices()),
                get_vector_from_repeated(homonym_query_request.dismissed_words()),
                static_cast<Vocinity::Homophonic_Alternative_Composer::Matching_Method>(
                    homonym_query_request.matching_method())};


            const bool is_levenshtein =
                static_cast<Vocinity::Homophonic_Alternative_Composer::Matching_Method>(
                    homonym_query_request.matching_method())
                == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
                    Phoneme_Levenshtein;

            const auto& homonym_model_configuration =
                is_levenshtein ? _model_configurations.at(model_code).second.second
                               : _model_configurations.at(model_code).second.first;

            auto composer = is_levenshtein
                                ? _homonym_composers.at(homonym_model_configuration.id).second
                                : _homonym_composers.at(homonym_model_configuration.id).first;


            const auto& combinations = run_homonoym_composing(input, composer, instructions);

            std::vector<std::string> queries;
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
                    const auto full_statement =
                        material.pre_context() + alternative + material.post_context();

#ifdef CPP17_AVAILABLE
                    std::transform(std::execution::unseq,
                                   full_statement.cbegin(),
                                   full_statement.cend(),
                                   full_statement.begin(),
                                   [](const auto& c) {
                                       return static_cast<char>(
                                           std::tolower(static_cast<unsigned char>(c)));
                                   });
#else
                    std::transform(full_statement.cbegin(),
                                   full_statement.cend(),
                                   full_statement.begin(),
                                   [](const auto& c) {
                                       return static_cast<char>(
                                           std::tolower(static_cast<unsigned char>(c)));
                                   });
#endif

                    queries.push_back(full_statement);
                }
            }

            if(not queries.empty())
            {
                auto scorer = _scorers.at(model_code);
                const auto& results =
                    do_scoring(queries, scorer, material.per_char_normalized());
                for(uint result_order = 0; result_order < results.size(); ++result_order)
                {
                    const auto& result = results.at(result_order);
                    auto score         = reply->add_scores();
                    score->set_input(queries.at(result_order));
                    score->set_production(result.production);
                    score->set_mean(result.mean);
                    score->set_g_mean(result.g_mean);
                    score->set_h_mean(result.h_mean);
                    score->set_negative_log_likelihood(result.negative_log_likelihood);
                    score->set_loss(result.loss);
                    score->set_sentence_probability(result.sentence_probability);
                }
            }

            return grpc::Status::OK;
        }

      private:
        std::vector<std::vector<std::vector<std::string>>> run_homonoym_composing(
            const std::string& input,
            std::shared_ptr<Vocinity::Homophonic_Alternative_Composer> composer,
            const Vocinity::Homophonic_Alternative_Composer::Instructions& instructions)
        {
            const auto& splitted_raw_sentences = akil::string::split(input, '.');
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
                auto chrono = std::chrono::high_resolution_clock::now();
                const auto word_combinations =
                    composer->get_alternatives(sentence, instructions, true);
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::high_resolution_clock::now() - chrono)
                                          .count();
                std::cout << "Homonym generation took " << duration << " msecs" << std::endl;
                std::cout << "----------------------------------------------------------------"
                             "--------"
                             "-----------------------------"
                          << std::endl;

                combinations[sentence_order].push_back(akil::string::split(sentence, ' '));
                for(ushort block_order = 0; block_order < word_combinations.size();
                    ++block_order)
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
                                                         (uint) block_order))
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

                std::cout << "----------------------------------------------------------------"
                             "--------"
                             "-----------------------------"
                          << std::endl;
            }
            return combinations;
        }

        template <typename T>
        static std::vector<T> get_vector_from_repeated(
            google::protobuf::RepeatedPtrField<T> repeated)
        {
            return std::vector<T>{repeated.cbegin(), repeated.cend()};
        }

        template <typename T>
        static std::span<T> get_span_from_repeated(
            google::protobuf::RepeatedPtrField<T> repeated)
        {
            return std::span<T>{repeated.cbegin(), repeated.cend()};
        }

        template <typename T>
        static std::vector<T> get_vector_from_repeated(
            google::protobuf::RepeatedField<T> repeated)
        {
            return std::vector<T>{repeated.cbegin(), repeated.cend()};
        }

        template <typename T>
        static std::span<T> get_span_from_repeated(google::protobuf::RepeatedField<T> repeated)
        {
            return std::span<T>{repeated.cbegin(), repeated.cend()};
        }

        std::vector<Vocinity::Context_Scorer::Score> do_scoring(
            const std::vector<std::string>& utterances,
            std::shared_ptr<Vocinity::Context_Scorer> scorer,
            const bool per_char_normalized)
        {
            if(utterances.empty())
            {
                return {};
            }

            if(utterances.size() == 1)
            {
                return {scorer->score_context(utterances.front(), per_char_normalized, false)};
            }

            return scorer->score_contexts(utterances, per_char_normalized);
        }

        bool create_processors(const std::string& model_code)
        {
            const std::lock_guard lock(_initialization_mutex);

            if(not _model_configurations.contains(model_code))
            {
                return false;
            }

            const auto& [model_configuration, homonym_composer_configuration] =
                _model_configurations.at(model_code);

            if(not _scorers.contains(model_code))
            {
                auto model_initialization_chrono = std::chrono::high_resolution_clock::now();

                auto scorer = std::make_shared<Vocinity::Context_Scorer>(
                    model_configuration.model_path,
                    model_configuration.type,
                    model_configuration.tokenizer_configuration,
                    model_configuration.precision,
                    model_configuration.hardware);
                // scorer->flush_cuda_tensor_cache_before_inference();
                scorer->optimize_parallelization_policy_for_use_of_multiple_instances();

                std::cout << "Instance of" << model_code << " model initialization took: "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::high_resolution_clock::now()
                                 - model_initialization_chrono)
                                 .count()
                          << " milliseconds\n\n";

                _scorers[model_code] = scorer;
            }

            if(not _homonym_composers.contains(
                   model_configuration.homonym_composer_configuration_id))
            {
                const auto& phonetics_dictionary =
                    Vocinity::Homophonic_Alternative_Composer::load_phonetics_dictionary(
                        homonym_composer_configuration.dictionary_path);

                auto composer = std::make_shared<Vocinity::Homophonic_Alternative_Composer>(
                    phonetics_dictionary);
                if(not homonym_composer_configuration.precomputed_phoneme_similarity_map
                           .empty())
                {
                    auto similarity_map = Vocinity::Homophonic_Alternative_Composer::
                        load_precomputed_phoneme_similarity_map(
                            homonym_composer_configuration.precomputed_phoneme_similarity_map);
                    composer->set_precomputed_phoneme_similarity_map(
                        std::move(similarity_map),
                        homonym_composer_configuration.is_levenshtein_dump);
                }

                _homonym_composers[homonym_composer_configuration.id] = composer;
            }
            return true;
        }

      private:
        static inline std::mutex _initialization_mutex;
        Unordered_Map<Model_Code, std::shared_ptr<Vocinity::Context_Scorer>> _scorers;
        Unordered_Map<uint32_t,
                      std::pair<std::shared_ptr<Vocinity::Homophonic_Alternative_Composer>,
                                std::shared_ptr<Vocinity::Homophonic_Alternative_Composer>>>
            _homonym_composers;
        Unordered_Map<std::string, std::vector<uint32_t>> _clients;
        Unordered_Map<Model_Code,
                      std::pair<Scorer_Model_Configuration,
                                std::pair<Phoneme_Based_Homonym_Composer_Configuration,
                                          Levenshtein_Homonym_Composer_Configuration>>>
            _model_configurations;
    };
};

int
main(int argc, char* argv[])
{
    setlocale(LC_NUMERIC, "C");
    const auto physical_cores = std::thread::hardware_concurrency() / 2;
    std::cout << physical_cores << " physical cores available." << std::endl;
    std::cout << argv[4] << " device is selected" << std::endl;

    const std::string phonetics_dictionary =
        "/opt/cloud/projects/vocinity/models/context-scorer/cmudict-0.7b.txt";
    if(false)
    {
        auto method =
            Vocinity::Homophonic_Alternative_Composer::Matching_Method::Phoneme_Transcription;
        if(method
           == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
               Phoneme_Transcription)
        {
            const auto similarity_map_composed = Vocinity::Homophonic_Alternative_Composer::
                precompute_phoneme_similarity_map_from_phonetics_dictionary(
                    phonetics_dictionary, 2, 0, false);
            Vocinity::Homophonic_Alternative_Composer::save_precomputed_phoneme_similarity_map(
                similarity_map_composed,
                "./similarity_map-cmudict07b-dist2-phoneme_transcription.cbor",
                true);
        }
        else if(method
                == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
                    Phoneme_Levenshtein)
        {
            const auto similarity_map_composed = Vocinity::Homophonic_Alternative_Composer::
                precompute_phoneme_similarity_map_from_phonetics_dictionary(
                    phonetics_dictionary, 2, 0, true);
            Vocinity::Homophonic_Alternative_Composer::save_precomputed_phoneme_similarity_map(
                similarity_map_composed,
                "./similarity_map-cmudict07b-dist2-phoneme_levenshtein.cbor",
                true);
        }

        return 0;
    }

    /////////////////////////////////////////////////////////////////////////

    const std::string host = "0.0.0.0";
    const std::string port = "1991";
    const std::string address(host + ":" + port);

    /////////////////////////////////////////////////////////////////////////

    Context_Scorer_Server::Homonym_Composer_Configuration
        homonym_pho_based_composer_configuration;
    homonym_pho_based_composer_configuration.id              = 0;
    homonym_pho_based_composer_configuration.dictionary_path = phonetics_dictionary;
    homonym_pho_based_composer_configuration.max_distance    = 2;
    homonym_pho_based_composer_configuration.matching_method =
        Vocinity::Homophonic_Alternative_Composer::Matching_Method::Phoneme_Transcription;
    homonym_pho_based_composer_configuration.precomputed_phoneme_similarity_map =
        "/opt/cloud/projects/vocinity/models/context-scorer/"
        "similarity_map-cmudict07b-dist2-phoneme_transcription.cbor";
    homonym_pho_based_composer_configuration.is_levenshtein_dump=false;

    Context_Scorer_Server::Homonym_Composer_Configuration
        homonym_lev_based_composer_configuration;
    homonym_lev_based_composer_configuration.id              = 0;
    homonym_lev_based_composer_configuration.dictionary_path = phonetics_dictionary;
    homonym_lev_based_composer_configuration.max_distance    = 2;
    homonym_lev_based_composer_configuration.matching_method =
        Vocinity::Homophonic_Alternative_Composer::Matching_Method::Phoneme_Levenshtein;
    homonym_lev_based_composer_configuration.precomputed_phoneme_similarity_map =
        "/opt/cloud/projects/vocinity/models/context-scorer/"
        "similarity_map-cmudict07b-dist2-phoneme_levenshtein.cbor";
    homonym_lev_based_composer_configuration.is_levenshtein_dump=true;



    Context_Scorer_Server::Scorer_Model_Configuration generic_model_configuration;
    generic_model_configuration.homonym_composer_configuration_id = 0;
    generic_model_configuration.model_path                        = argv[1];
    generic_model_configuration.hardware =
        std::string(argv[4]) == "--cuda" ? Vocinity::Context_Scorer::Inference_Hardware::CUDA
                                         : Vocinity::Context_Scorer::Inference_Hardware::CPU;
    generic_model_configuration.precision = Vocinity::Context_Scorer::Precision::FP32;
    generic_model_configuration.tokenizer_configuration =
        Vocinity::Context_Scorer::Tokenizer_Configuration{argv[2], argv[3]};
    generic_model_configuration.type = Vocinity::Context_Scorer::GPT_TYPE::DistilGPT2;



    Unordered_Map<Context_Scorer_Server::Model_Code,
                  std::pair<Context_Scorer_Server::Scorer_Model_Configuration,
                            std::pair<Context_Scorer_Server::Phoneme_Based_Homonym_Composer_Configuration,
                                      Context_Scorer_Server::Levenshtein_Homonym_Composer_Configuration>>> configuration;
    configuration["generic"]={generic_model_configuration,{homonym_pho_based_composer_configuration,
                        homonym_lev_based_composer_configuration}};

    Context_Scorer_Server::Service service;
    service.initialize(std::move(configuration));

    /////////////////////////////////////////////////////////////////////////

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());

    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on port: " << address << std::endl;

    server->Wait();

    return 0;
}
