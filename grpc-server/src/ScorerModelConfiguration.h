#ifndef SCORER_MODEL_CONFIGURATION_H_
#define SCORER_MODEL_CONFIGURATION_H_

#include <filesystem>
#include "../src/Context-Scorer.hpp"

struct ScorerModelConfiguration
{
    uint32_t homonym_composer_configuration_id = 0;
    std::filesystem::path model_path;
    Vocinity::Context_Scorer::GPT_TYPE type =
        Vocinity::Context_Scorer::GPT_TYPE::DistilGPT2;
    Vocinity::Context_Scorer::Tokenizer_Configuration tokenizer_configuration;
    Vocinity::Context_Scorer::Precision precision =
        Vocinity::Context_Scorer::Precision::FP16;
    Vocinity::Context_Scorer::Inference_Environment environment =
        Vocinity::Context_Scorer::Inference_Environment::CUDA;
};

#endif
