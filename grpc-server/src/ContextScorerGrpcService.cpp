#include "ContextScorerGrpcService.h"

void ContextScorerGrpcService::initialize(robin_hood::unordered_flat_map<ModelCode,
		std::pair<ScorerModelConfiguration, HomonymComposerConfiguration> > &&configuration,
		const std::string generic_model_code)
{
	_model_configurations = std::move(configuration);
	_generic_model_code   = generic_model_code;

	create_processors(_generic_model_code);
}

grpc::Status ContextScorerGrpcService::say_hi(grpc::ServerContext *context,
		const Vocinity::Knock_Knock *request, Vocinity::Nothing *)
{
	const uint32_t client_id =
			akil::functional::getDefinitelyUniqueId<uint32_t>("grpc_client_id");
	const auto wanted_models_size = request->models_that_you_planning_to_use_size();
	for(int model_order = 0; model_order < wanted_models_size; ++model_order)
	{
		const auto wanted_model =
				request->models_that_you_planning_to_use(model_order);
		const std::string model_code =
				wanted_model.code().empty() ? _generic_model_code : wanted_model.code();

		if(not _model_configurations.contains(model_code))
		{
			return grpc::Status(grpc::StatusCode::NOT_FOUND,
								"There is no such model as: " + model_code);
		}

		create_processors(model_code);
	}

	return wanted_models_size
			? grpc::Status::OK
			: grpc::Status(grpc::StatusCode::NOT_FOUND,
						   "You should subscribe at least one model");
}

grpc::Status ContextScorerGrpcService::get_homonyms(grpc::ServerContext *context,
		const Vocinity::Homonym_Generation_Query *request, Vocinity::Homonyms *reply)
{
	const std::string model_code =
			request->model_code().empty() ? _generic_model_code : request->model_code();

	if(not _model_configurations.contains(model_code))
	{
		return grpc::Status(grpc::StatusCode::NOT_FOUND,
							"There is no such model as: " + model_code);
	}

	const auto& input = request->input();

	const auto& instructions = Vocinity::Homophonic_Alternative_Composer::Instructions{
			(ushort) request->max_num_of_best_homophonic_alternatives(),
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
			_model_configurations.at(model_code).second;

	auto composer = _homonym_composers.at(homonym_model_configuration.id);

	std::vector<std::vector<std::vector<std::string>>> combinations;
	try
	{
		combinations = run_homonoym_composing(input, composer, instructions);
	}
	catch(const std::exception& e)
	{
		return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
	}

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

grpc::Status ContextScorerGrpcService::get_best_n_alternatives(grpc::ServerContext* context,
		const Vocinity::Context_Scoring_Query* request, Vocinity::Context_Score* reply)
{
	auto chrono                      = std::chrono::high_resolution_clock::now();
	const auto& material             = request->material();
	const auto homonym_query_request = material.input();
	const std::string model_code     = homonym_query_request.model_code().empty()
			? _generic_model_code
			: homonym_query_request.model_code();

	if(not _model_configurations.contains(model_code))
	{
		return grpc::Status(grpc::StatusCode::NOT_FOUND,
							"There is no such model as: " + model_code);
	}

	const auto& input = homonym_query_request.input();

	const auto& instructions = Vocinity::Homophonic_Alternative_Composer::Instructions{
			(ushort) homonym_query_request.max_num_of_best_homophonic_alternatives(),
			(short) homonym_query_request.max_distance(),
			get_vector_from_repeated(homonym_query_request.dismissed_word_indices()),
			get_vector_from_repeated(homonym_query_request.dismissed_words()),
			static_cast<Vocinity::Homophonic_Alternative_Composer::Matching_Method>(
				homonym_query_request.matching_method())};

	//			const bool is_levenshtein =
	//			    static_cast<Vocinity::Homophonic_Alternative_Composer::Matching_Method>(
	//			        homonym_query_request.matching_method())
	//			    == Vocinity::Homophonic_Alternative_Composer::Matching_Method::
	//			        Phoneme_Levenshtein;

	const auto& homonym_model_configuration =
			_model_configurations.at(model_code).second;

	auto composer = _homonym_composers.at(homonym_model_configuration.id);
	std::vector<std::vector<std::vector<std::string>>> combinations;
	try
	{
		combinations = run_homonoym_composing(input, composer, instructions);
	}
	catch(const std::exception& e)
	{
		return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
	}

	std::vector<std::string> alternatives;
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
			alternatives.push_back(alternative);
			std::string full_statement = alternative;
			if(not material.pre_context().empty())
			{
				full_statement = material.pre_context() + " " + full_statement;
			}
			if(not material.post_context().empty())
			{
				full_statement += " " + material.post_context();
			}

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
		std::vector<Vocinity::Context_Scorer::Score> results;
		try
		{
			results = do_scoring(queries, scorer, material.per_char_normalized());
		}
		catch(const std::exception& e)
		{
			return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
		}

		for(ushort result_order = 0; result_order < alternatives.size();
			++result_order)
		{
			results[result_order].utterance = alternatives.at(result_order);
		}

#ifdef CPP17_AVAILABLE
		std::sort(std::execution::unseq,
				  results.begin(),
				  results.end(),
				  [](const auto& one, const auto& another) -> bool
		{ return one.mean > another.mean; });
#else
		std::sort(results.begin(),
				  results.end(),
				  [](const auto& one, const auto& another) -> bool
		{ return first_score.mean > second_score.mean; });
#endif
		for(uint result_order = 0; result_order < results.size(); ++result_order)
		{
			const auto& result = results.at(result_order);
			auto score         = reply->add_scores();
			score->set_input(result.utterance);
			score->set_production(result.production);
			score->set_mean(result.mean);
			score->set_g_mean(result.g_mean);
			score->set_h_mean(result.h_mean);
			score->set_negative_log_likelihood(result.negative_log_likelihood);
			score->set_loss(result.loss);
			score->set_sentence_probability(result.sentence_probability);
		}
	}

	const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - chrono)
			.count();
	std::cout << "Scoring took " << duration << " msecs" << std::endl;

	return grpc::Status::OK;
}
