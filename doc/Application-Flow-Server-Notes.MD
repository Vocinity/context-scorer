# Aplication Flow and Server Notes
## In the beginning...
### OS Deps
* Apt
```bash
sudo apt install build-essential git qt5-qmake qtbase5-dev-tools ccache libtbb-dev libpython3-dev g++-10 libsl-dev libcrypto++-dev
```
* Setup default gcc and g++
```bash
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```
if:
```bash
update-alternatives: error: no alternatives for gcc
update-alternatives: error: no alternatives for g++
```
then:
```bash
#no alternatives for gcc
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 20
#no alternatives for g++
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20
```
again `update-alternatives --config`.
> Having g++-11 from ppa:ubuntu-toolchain-r/test with [package pinning](https://gist.github.com/JPvRiel/8ae81e21ce6397a0502fedddca068507) would be nice.

### apt-rd Deps

* [grpc-127-dev](https://github.com/Vocinity/apt-rd#grpc-127-dev)
* [protobuf-suite-dev](https://github.com/Vocinity/apt-rd#protobuf-suite-dev)

### Other *External* Deps
* aMisc will point context-scorer for external deps by [qmake/depend_aMisc_template.pri](https://github.com/Vocinity/aMisc/blob/stable/qmake/depend_aMisc_template.pri) 
* `USE_TORCH_CUDA_RT` is not needed for libtorch 1.10 and onnx forced cuda 11.4 stack.

## Homonym Generation
### Terminology
means producing alternative sentences by combining spelling variations of each word in original sentence.

Spelling characteristics of words are encoded by phonemess.

> What a phoneme looks  like? You can see thousands of them there: https://drive.google.com/file/d/1XXOsqJkRb2DyAmA284rkQQXQozgppQMX/view?usp=sharing

We will be mentioning dictionary concept by examples over and over while describing each of these similarity computation methods, dont worry, *just stay with me*. 

Variations are computed by:
* `phoneme_transcription`: counting missing, extra or different phonemes
* `phoneme_levenshtein`: not counting phoneme difference but calculating character difference of phonemes

up to a `distance` and up to `max_num_of_best_homophonic_alternatives`count.

#### phoneme_transcription

We have a raw english `query_word="smart"`and we want `max_num_of_best_homophonic_alternatives=5` homophonic alternatives up to `distance=2` 

1. Go phoneme dictionary find phonemes of "smart": S M AA1 R T. If you can not find, then synthesize using embedded deep grapheme to phoneme converter.
2. Walk over whole dictionary and find raw english words which have at most 2 different, missing or extra phonemes
3. Order findings by `distance`
4. Keep best 5 `max_num_of_best_homophonic_alternatives`.
5. Tell me result.

```
SMARTT (~0) | S M AA1 R T  | perfect homonym, spelled exactly like "smart"
START (~1)  | S T AA1 R T  | 1 different phoneme
MAHRT (-1)  |   M AA1 R T  | 1 missing phoneme
MART (-1)   |   M AA1 R T  | 1 missing phoneme (same with the previous one)
MARTE (-1)  |   M AA1 R T  | 1 missing phoneme (same with the previous ones)
```
##### Similarity Map Concept

We are cool engineers (because we have 50 msecs in total for everything) and of course we are not looping over whole dictionary again and again everytime.
Rather we are keeping precomputed distance lookup table of dictionary words for `phoneme_transcription` matching method. (Because I dropped others.)

So cool version of previous steps:
1. Load similarity map big enough to find `distance=2` (precomputed up to distance=2)
2. Random access word "smart". If you can not, report it to be added asynchronously in background, loop over whole dictionary like before.
3. Get best 5 homonyms of it.

difference would be seconds...

#### phoneme_levenshtein

Everything is same. Except we are getting levenshtein distance (character difference, commonly used to calculate Word Error Rate) SMAA1RT,STAA1RT,MAA1RT strings. There are some obsolote reasons to have this method. Will be dropped soon.

### Homonym Generation Flow
> It is expected to have big changes in this section after https://github.com/Vocinity/context-scorer/issues/1 resolved.

`Homophonic_Alternative_Composer` constructor accepts <raw english,phonemes> word pairs of dictionary and you can use
`load_phonetics_dictionary` parser fore cmudict-like text file dictionaries.

At this point you are done in primitive bad-latency way.

To import pre-computed similarity maps you should use `load_precomputed_phoneme_similarity_map` helper and `set_precomputed_phoneme_similarity_map` setter.
Now you are done in cool-way.

 Similariy maps are produced by `precompute_phoneme_similarity_map_from_phonetics_dictionary` and saved by `save_precomputed_phoneme_similarity_map`.

*Matching methods out of `Matching_Method::Phoneme_Transcription` are obsolote, we will keep them but wont use, so no need to document.*

Please see [Client document](https://github.com/Vocinity/context-scorer/blob/stable/doc/Usage-Instructions-Client-Notes.MD) for further explanation about `Homophonic_Alternative_Composer::Instructions`.

And finally:
```cpp
Alternative_Words_Of_Sentence get_alternatives(const std::string& reference,
                                               const Instructions& instructions,
                                               const bool parallel = false);
```
is way to go.

## Context Scorer
### Terminology 

* `Inference_Environment`: Refers the way we utilize the devices that we run deep learning models. `TensorRT` and `CUDA` are two separate ways to use NVIDIA GPU's.
`CPU` option means either Libtorch backend or ONNX optimized CPU inference depending compile time availability.
* `Precision`: `FP32`is full precision slow float mode. In case of `FP16`, some/all weights can be stored & processed purely/partially in 2 bytes uint16_t as half/mixed precision depending model. 
* `GPT_TYPE`: Name of the model from OPENAI or EleutherAI HuggingFace GPT releases.
* Vocabulary and "Merges" Files: Tokenizer material. Will be published alongside gpt models.
* bos, eos, pad, unk, mask tokens are simply numerical codes for those situations in tokenization. It will be mentioned when new model released if any of those tokenizer parameters differs.
* `parallelization_policy` inter and intra threading policy for torch tensor operations.
* `scorer_model_path` pt/onnx file.
* `context` is a string. pre-context, input query and post-context are parts of the context string.
* `per_char_normalized` divides and normalizes score by context string character size.
* `intra-batching` term refers batching chunks of single context in contrast of batching multiple contexts.
* `max_sequence_length` is length of string that model can process at once.
* `flush_cuda_tensor_cache_before_inference` makes torch cuda tensor op cache cleaned-up before each inference to save GPU memory more than we should.

### Context Scoring Flow
#### Before

* Good to call `optimize_parallelization_policy_for_use_of_multiple_instances`. Having single thread for torch ops is
almost always better for us.
* Good to decide batch size(s) if you are planning to use `TensorRT` inference environment. You can either have a
constant, big enough batch size and and fill with dummy items when you have fewer elements to process, or you can
employ multiple instances with multiple
batch sizes if there is enough gpu mem. Reason to do so is time & memory cost of recreating TensorRT engine cache
for different tensor shapes.
So do that at the beginning, not in runtime while client wants you to respond ASAP.

#### Instantiation

```cpp
explicit Context_Scorer(const std::filesystem::path& scorer_model_path,
                        const GPT_TYPE type = GPT_TYPE::DistilGPT2,
                        const Tokenizer_Configuration& encoding_conf = {},
                        const Precision precision                    = Precision::FP32
#ifdef CUDA_AVAILABLE
                        ,
                        const Inference_Environment environment = Inference_Environment::CPU
#endif
);
```
* `scorer_model_path`is path to pt or onnx file. If filename.onnx.data file is shared with you, then you
SHOULD NOT CHANGE filename.onnx.data filename and put this file next to onnx file.
* `type` information will be given and default argument does not work for all.
* `encoding_conf` contains the configuration for the tokenizer. Default construction would work unless
stated otherwise.
* `precision` is up to you. `FP16` precision is always faster if such a model shared for this `type`.
Currently you can use FP32 for FP16 models (this can change). Opposite does not make sense out of debugging purposes. 
* `environment` can be `TensorRT` if you are aware of the engine cache latency that occurs in
initialization and variable inference batch size where TensorRT needs to cache given tensor shape again and again.
`CUDA` and `CPU` environments dont have such an issue.

#### Query

* Currently sentences in context should end with the `.` in pre-context (if not empty), input variations and post-context
(if not empty).
* Currently whole context string should be normalized to lowercase letters.

```cpp
/**
 * @brief consider_intra_batching=true allows intra batching of one long single context by
 * dispatching parts of it once stacked as a batch.
 *
 * this is a good idea for decent gpu and long text but here the point:
 * -Especially in TensorRT, if next time you run this function for
 * different multiplier of get_max_sequence_length characters (I hardcoded 64,
 * optimal max is 1024) then graph optimizer needs to profile that new different
 * shape of dynamic axis and should update engine cache.
 * This means you will lose seconds at the beginning of next run.
 * But if you split your sequential runs as equal length of 64 chars of blocks
 * or you are not planning to run this function again then no problem.
 * Note that, scores are only reproducible for same batch set.
 *
 *
 * consider_intra_batching=false has no such constrain and slower
 * if text is too long and your gpu is not saturated. It just runs
 * get_max_sequence_length chars long padded blocks of context one by one sequentially
 * in sliding window manner without batching. So input is always in the
 * [1,get_max_sequence_length] shape.
 * Summing up results of two blocks and inferencing two blocks at once is same for us
 * in terms of accuracy in our way of perplexity computation.
 *
 */
Score score_context(const std::string& context,
                    const bool per_char_normalized     = true,
                    const bool consider_intra_batching = false);
/**
 * @brief is batching perplexity computation of multiple separate contexts.
 * Note that, scores are only reproducible for same batch set.
 *
 * Scores will be similar for same item between single and batch runs but not
 * same.
 *
 * Result vector is in same order with the contexts vector/
 */
std::vector<Score> score_contexts(const std::vector<std::string>& contexts,
                                  const bool per_char_normalized = true);
```
* First run always will be slow. In `TensorRT`environment, extremely slow. So you should always warmup your instance
right after you construct the object before actual runtime.
* Long context and/or many contexts at once will cost more time and memory. Watch out your gpu memory.

> Read [Client Instructions](https://github.com/Vocinity/context-scorer/blob/stable/doc/Usage-Instructions-Client-Notes.MD)
contract again.

## Serving

[@sind4l](https://github.com/sind4l) will describe.
