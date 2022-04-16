# Usage Instructions and Client Notes
## Journey
### Hi
Runtime starts with a "Knock_Knock" query. 
You should "say_hi" as early as possible to let server to initialize and warmup the models that you are planning to use. This is a blocking request for you.
> Note that initializations of a model can even take up to a minute in case of TensorRT inference backend.

Having no model code to subscribe will tell server to initialize `generic`model. Likely there will be multiple models for different domains.
> Your previously initialized model may die after some time to have room in gpu memory to be able to initialize other models. So periodically saying hi is recommended.
### Query Preparation
```python
@dataclasses.dataclass
class Homonym_Generation_Query:
    input: str
    max_num_of_best_homophonic_alternatives: int
    max_distance: int
    dismissed_word_indices: List[int]
    dismissed_words: List[str]
    model_code: str = ""


@dataclasses.dataclass
class Context_Scoring_Query:
    pre_context: str
    input: Homonym_Generation_Query
    post_context: str
    per_char_normalized: bool
```
Context_Scoring_Query query wraps Homonym_Generation_Query plus supportive pre or post context to score.

You can just get available homonyms and decide to not to score that context if homophonic alternatives seem useless to you. But probably you wont have any such heuristics to judge homonyms. 
> Context_Scoring_Query always computes homophonic alternatives each time, you can not reuse response of Homonym_Generation_Query but ideally homonym lookup should be super fast enough to not to care lackness of reusability.

#### Homonym_Generation_Query

##### `input`
is what you want to produce sentences as permutations of homophonic relatives of EACH input WORD.

* Word is space separated substring of this string.
* What you put at the end like `? or ! or . or ,` does not matter. Uppercase or lowercase dont matter.
* Apostrophe matters for homonym lookup. Please visit Homonym Dictionary for it.
* Abbrevitions matter for homonym lookup. You know where to visit for notion. 
* Shorter input means less work and less room for errors. Keep as short as possible.

##### `max_num_of_best_homophonic_alternatives`
is UPPER limit of number of homonyms to produce. 
* There may be less than `max_num_of_best_homophonic_alternatives` available with given distance. Even maybe none.
* Keeping this number low is fast and safe just like input length.

##### `max_distance`
is UPPER limit of allowed phoneme difference distance.

* Similarity map database preparation defines actual upper limit for distance you can travel.
Currently `distance=2`is feasible but as long as you can afford such a giant cartesian dictionary it is okay to host bigger distances.
`max_num_of_best_homophonic_alternatives`will limit your lookup path starting by `distance=0` aka perfect homonyms.

##### `dismissed_word_indices`
is zero based array indice of words that you dont want server to produce alternatives. Original word will be used for context scoring.

##### `dismissed_words`
is list of words that you dont want server to produce alternatives. Original word will be used for context scoring.

##### [obsolote]`matching_method`
There are 2 homonym matching methods. Planned to be dropped.

##### `model_code`
which scorer model will score these produced homonyms. Empty is fine (means generic).

Having scorer model id in Homonym_Generation_Query may seem unintutive but think mode_code like context_code.
"Ideally" models are trained for the context that we build our homonyms dictionary. So this code is defining, model, homonym dictionary and relevant domain all together.

Please see `preset` parameter below for complete picture.

---
For example "Smart rower." produces below combinations with the `max_distance=2`,` max_num_of_best_homophonic_alternatives=5`:
```
SMARTT (~0)
START (~1)
MAHRT (-1)
MART (-1)
MARTE (-1)
        ROHER (~0)
        ROLLE (~1)
        ROEHL (~1)
        ROLL (~1)
        ROLE (~1)

~ is difference.
- is have n less phonemes.
+ means having n more phonemes.
```

#### Context_Scoring_Query

##### `pre_context`
is the string prepended to each alternative sentence for scoring.
* More context=better accuracy. Unnecessary context=worse accuracy.

##### `input`
see above

##### `post_context`
is the string appended to each alternative sentence for scoring.
* More context=better accuracy. Unnecessary context=worse accuracy.

##### `per_char_normalized`
do you want me to normalize score for sequence length? Yes it matters. Being lengthy or not is changing score alone.

##### [not implemented yet but use]`preset`
You can order server to be quick or accurate. There are things that context-scorer_server can do according your needs. Simply server can run this inference using smaller or deeper model trained for your context.


---

For example:
```
pre_context:`Click on the eye in the icon tray to pick your product of interest or say echelon-connect bike or smart rower.`
input: `Smart rower.`
post context: nothing.
```
* produced homophonic relative of input: `SMART ROWER.` (original is its own perfect homonym and will be scored too. Even mostly it is correct alternative.)
* produced homophonic relative of input: `START ROHER.`
* produced homophonic relative of input: `SMART ROHER.`

> Dot will be appended to alternative sentences by [server](https://github.com/Vocinity/context-scorer/blob/9fa0d26ed421a2ab2c1b1dc2599995d0928d1f45/grpc-server/src/main.cpp#L229). (still thinking about others like ?!) 


What scoring sees:
```
Click on the eye in the icon tray to pick your product of interest or say echelon-connect bike or smart rower. smart rower.
Click on the eye in the icon tray to pick your product of interest or say echelon-connect bike or smart rower. start roher.
click on the eye in the icon tray to pick your product of interest or say echelon-connect bike or smart rower. smart roher.
...
```
### Scores
You will get results ordered by `mean` score and best is greatest is first. Lets see an example of scoring `echelon collect` input:
```
(venv) isgursoy@isgursoy-VORKE:/opt/cloud/projects/vocinity/context-scorer/grpc-client$ python3 ./client.py 
input: "click on the eye in the icon tray to pick your product of interest or say echelon connect bike or smart rower. echelon connect."
production: -4.998478608972886
mean: 0.0044642038205090695
g_mean: 0.00014522241647629177
h_mean: -0.2950199912576114
negative_log_likelihood: 8.371323529411764
loss: 163.7988999310662
sentence_probability: -4.010920718834093e-122

input: "click on the eye in the icon tray to pick your product of interest or say echelon connect bike or smart rower. echelon exit."
production: -5.225337308995864
mean: 0.004214285489390878
g_mean: 0.00011411755729247542
h_mean: -0.2979620204252355
negative_log_likelihood: 8.371323529411764
loss: 163.7988999310662
sentence_probability: -4.010920718834093e-122

input: "click on the eye in the icon tray to pick your product of interest or say echelon connect bike or smart rower. echelon collect."
production: -5.18399855669807
mean: 0.00421385423225515
g_mean: 0.00011924158453064807
h_mean: -0.2952853932100184
negative_log_likelihood: 8.371323529411764
loss: 163.7988999310662
sentence_probability: -4.010920718834093e-122
```
* All scores have a meaning but `mean` is most robust and easy to interpret. 

* Normally perplexity score is used together with the beam search decoder like:
```
final_score = beam_search_score + rescorer_alpha*neural_rescorer_score + rescorer_beta*seq_length
```
But we dont have a decoder here and yes, context scoring is more error prone than usual. In this case you can use ASR provider's probability to  exempt high-confidence words from homonym lookup.

* We can not use a threshold to assume >x is good. Because we dont know characteristics & training knowledge of speech encoder and decoder configuration. So searching for a "works" threshold over a validation set is not much meaningful.

## Notes
* Server may respond your query with `INTERNAL` type grpc exception because of a c++ crash. This is just for notification.
"Ideally" server should log it, restart itself, reinitialize previously subscribed models and recover runtime properly.
In such case you should loop (server restart plus model reinitialization will take time, have some sleep, dont burn your cpu)
say_hi request until not getting `StatusCode.UNAVAILABLE` response after server is up and your next say_hi will return OK when server is done with the reinitialization procedure.
