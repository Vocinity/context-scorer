#ifndef SERVICE_H_
#define SERVICE_H_

#include "context-scorer.grpc.pb.h"
#include "context-scorer.pb.h"
#include "ScorerModelConfiguration.h"
#include "HomonymComposerConfiguration.h"

class ContextScorerGrpcService final : public Vocinity::Context_Scorer_gRPC::Service
{
public:
	using ModelCode = std::string;

	void initialize(Unordered_Map<ModelCode,
			std::pair<ScorerModelConfiguration, HomonymComposerConfiguration>>&& configuration,
			const std::string generic_model_code = "generic");

	grpc::Status say_hi(grpc::ServerContext* context, const Vocinity::Knock_Knock* request, Vocinity::Nothing*) override;

	virtual grpc::Status get_homonyms(grpc::ServerContext* context, const Vocinity::Homonym_Generation_Query* request,
			Vocinity::Homonyms* reply) override;

	virtual grpc::Status get_best_n_alternatives(grpc::ServerContext* context,
		const Vocinity::Context_Scoring_Query* request, Vocinity::Context_Score* reply) override;

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
			const auto& word_combinations =
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
				model_configuration.precision
#ifdef CUDA_AVAILABLE
				,
				model_configuration.environment
#endif
			);
			// scorer->flush_cuda_tensor_cache_before_inference();
			scorer->optimize_parallelization_policy_for_use_of_multiple_instances();

			std::cout << "Instance of " << model_code << " model initialization took: "
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
	Unordered_Map<ModelCode, std::shared_ptr<Vocinity::Context_Scorer>> _scorers;
	Unordered_Map<uint32_t, std::shared_ptr<Vocinity::Homophonic_Alternative_Composer>>
			_homonym_composers;
	Unordered_Map<ModelCode,
			std::pair<ScorerModelConfiguration, HomonymComposerConfiguration>>
			_model_configurations;
	std::string _generic_model_code;
};

#endif
