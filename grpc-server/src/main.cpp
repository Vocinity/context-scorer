#include <thread>
#include <grpc++/grpc++.h>
#include "ContextScorerGrpcService.h"
#include "HomonymComposerConfiguration.h"
#include "ScorerModelConfiguration.h"

int main(int argc, char* argv[])
{
    setlocale(LC_NUMERIC, "C");
    const auto physical_cores = std::thread::hardware_concurrency() / 2;
    std::cout << physical_cores << " physical cores available." << std::endl;
    std::cout << argv[4] << " device is selected" << std::endl;

    const std::string phonetics_dictionary =
        "/opt/models/context-scorer/homonym-generator/cmudict-0.7b.txt";
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

    const std::string host = argv[1];
    const std::string port = argv[2];
    const std::string address(host + ":" + port);

    /////////////////////////////////////////////////////////////////////////

	HomonymComposerConfiguration homonym_pho_based_composer_configuration;
    //  homonym_pho_based_composer_configuration.id              = 0;
    homonym_pho_based_composer_configuration.dictionary_path = phonetics_dictionary;
    homonym_pho_based_composer_configuration.max_distance    = 2;
    homonym_pho_based_composer_configuration.matching_method =
        Vocinity::Homophonic_Alternative_Composer::Matching_Method::Phoneme_Transcription;
    homonym_pho_based_composer_configuration.precomputed_phoneme_similarity_map =
        "/opt/models/context-scorer/homonym-generator/"
        "similarity_map-cmudict07b-dist2-phoneme_transcription.cbor";
    homonym_pho_based_composer_configuration.is_levenshtein_dump = false;

    //    Context_Scorer_Server::Homonym_Composer_Configuration
    //        homonym_lev_based_composer_configuration;
    //    homonym_lev_based_composer_configuration.id              = 0;
    //    homonym_lev_based_composer_configuration.dictionary_path = phonetics_dictionary;
    //    homonym_lev_based_composer_configuration.max_distance    = 2;
    //    homonym_lev_based_composer_configuration.matching_method =
    //        Vocinity::Homophonic_Alternative_Composer::Matching_Method::Phoneme_Levenshtein;
    //    homonym_lev_based_composer_configuration.precomputed_phoneme_similarity_map =
    //        "/opt/cloud/projects/vocinity/models/context-scorer/"
    //        "similarity_map-cmudict07b-dist2-phoneme_levenshtein.cbor";
    //    homonym_lev_based_composer_configuration.is_levenshtein_dump=true;

    Vocinity::Context_Scorer::Inference_Environment environment;
    if(std::string(argv[6]) == "--cuda")
    {
        environment = Vocinity::Context_Scorer::Inference_Environment::CUDA;
    }
    else if(std::string(argv[6]) == "--cpu")
    {
        environment = Vocinity::Context_Scorer::Inference_Environment::CPU;
    }
    else if(std::string(argv[6]) == "--trt")
    {
        environment = Vocinity::Context_Scorer::Inference_Environment::TensorRT;
    }

	ScorerModelConfiguration generic_model_configuration;
    generic_model_configuration.homonym_composer_configuration_id =
        homonym_pho_based_composer_configuration.id;
    generic_model_configuration.model_path  = argv[3];
    generic_model_configuration.environment = environment;
    generic_model_configuration.precision   = Vocinity::Context_Scorer::Precision::FP16;
    generic_model_configuration.tokenizer_configuration =
        Vocinity::Context_Scorer::Tokenizer_Configuration{argv[4], argv[5]};
    generic_model_configuration.type = Vocinity::Context_Scorer::GPT_TYPE::DistilGPT2;


	Unordered_Map<ContextScorerGrpcService::ModelCode,
				  std::pair<ScorerModelConfiguration,
							HomonymComposerConfiguration>>
        configuration;
    configuration["generic"] = {generic_model_configuration,
                                homonym_pho_based_composer_configuration};

	ContextScorerGrpcService service;
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
