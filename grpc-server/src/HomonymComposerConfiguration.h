#ifndef HOMONYM_COMPOSER_CONFIGURATION_H_
#define HOMONYM_COMPOSER_CONFIGURATION_H_

#include <filesystem>
#include "../src/Homophonic-Alternatives.hpp"

struct HomonymComposerConfiguration
{
    HomonymComposerConfiguration()
    : max_distance(0)
    , id(akil::functional::getDefinitelyUniqueId<uint32_t>("homonym_composer_id"))
    , is_levenshtein_dump(false)
    {
    }

    std::filesystem::path dictionary_path;
    std::filesystem::path precomputed_phoneme_similarity_map;
    ushort max_distance;
    Vocinity::Homophonic_Alternative_Composer::Matching_Method matching_method;
    uint32_t id;
    bool is_levenshtein_dump;
};

#endif
