import os
import sys
import torch
import numpy as np
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
import inspect, time, math, json
import onnxruntime


np.set_printoptions(threshold=np.inf)

working_directory=os.getcwd()

# for torchscript tracing;
#replace GPT2LMHeadModel.forward arguments order in
# venv/lib/python3.8/site-packages/transformers/models/gpt2/modelling_gpt2.py#924
#as:
#
#def forward(
#        self,
#        input_ids=None,
#        attention_mask=None,
#        position_ids=None,
#        past_key_values=None,
#        labels=None,...

def process_labels(labels, logits):
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def encode(text,tokenizer,max_seq_length,pad_id):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    actual_token_size = len(ids)

    tokens_padded = ids.copy()
    if hasattr(tokenizer, "bos_token_id"):
        tokens_padded = [tokenizer.bos_token_id] + tokens_padded
    if hasattr(tokenizer, "eos_token_id"):
        tokens_padded = tokens_padded + [tokenizer.eos_token_id]
    padded_token_size = len(tokens_padded)

    full_sequence_size = max_seq_length - (padded_token_size % max_seq_length) + padded_token_size if (
                    padded_token_size % max_seq_length) else padded_token_size
    full_sequence = [pad_id] * full_sequence_size
    full_sequence[: padded_token_size] = tokens_padded

    input_mask = np.zeros(full_sequence_size)
    input_mask[: padded_token_size] = 1

    return torch.from_numpy(np.array(full_sequence)), torch.from_numpy(input_mask), actual_token_size

def export_torchscript(model,input,quantize=True,trt=False,onnx=False,device="cpu"):
    def conv1d_to_linear(model):
        from transformers.modeling_utils import Conv1D
        def _conv1d_to_linear(module):
            in_size, out_size = module.weight.shape
            linear = torch.nn.Linear(in_size, out_size)
            linear.weight.data = module.weight.data.T.contiguous()
            linear.bias.data = module.bias.data
            return linear
        for name in list(model._modules):
            module = model._modules[name]
            if isinstance(module, Conv1D):
                linear = _conv1d_to_linear(module)
                model._modules[name] = linear
            else:
                conv1d_to_linear(module)

    if quantize and device=="cpu":
        conv1d_to_linear(model)
        dynamic = True
        if dynamic:
            model=torch.quantization.quantize_dynamic(model=model, qconfig_spec={torch.nn.Linear},
                                                dtype=torch.qint8,
                                                inplace=False)
        else:
            perChannel = False
            model.qconfig = torch.quantization.get_default_qconfig(
                ('fbgemm' if args.device == "cpu" else "") if perChannel else "")
            print(model.qconfig)
            model=torch.quantization.prepare(model, inplace=False)
            with torch.no_grad():
                model(*input)
            model=torch.quantization.convert(model, inplace=False)

    model = torch.jit.trace(model,input)
    model.eval()

    if onnx:
        ONNX_FILE_PATH = working_directory+'/model.onnx'
        torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=['input'],
                          output_names = ['output'], export_params = True)
        model = onnx.load(ONNX_FILE_PATH)
        onnx.checker.check_model(model)

    if trt and device=="cuda":
        if t2trt:
            from torch2trt import torch2trt, TRTModule
            model_trt = torch2trt(model, [input])
            torch.save(model_trt.state_dict(),working_directory+"/traced_" + lm_model_file + "_"+device+".pt")
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(working_directory+"/traced_" + lm_model_file + "_"+device+".pt"))
        else:
            import trtorch
            compile_settings = {
                "inputs": [trtorch.Input(
                    shape=input[0].shape,
                    dtype=torch.int32,
                ),trtorch.Input(
                    shape=input[1].shape,
                    dtype=torch.int32,
                )],
                "truncate_long_and_double": True,
                "enabled_precisions": {
                    #torch.half
                    #torch.int8
                    torch.float
                },
            }
            model = trtorch.compile(model, compile_settings)
    return model

def nemo_gpt2_test(sentence, lm_model_file,device = "cuda",trace_model = False,support_att_mask=None,
                   hf_inference=False,warmup=False,onnx_runtime=True):
    print(onnx_runtime)
    hidden_size=768 #n_embd
    num_attention_heads=12#n_head
    num_layer=6#n_layer (hidden layers)
    label_ignore_id = -100
    if not trace_model and not hf_inference and not onnx_runtime:
        support_att_mask = True if support_att_mask is None else support_att_mask
        max_seq_length = 1024
        pad_id=50256

        model = torch.jit.load(working_directory+"/traced_" + lm_model_file + "_"+device+".pt").to(device)
        model_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=lm_model_file)
        model.eval()
    else:
        if onnx_runtime:
            onnx_model_path = "/opt/cloud/projects/vocinity/models/context-scorer/openai/distilgpt2-hf-onnx/distilgpt2_cuda_o1_int64_fp32.onnx"
            session_options=onnxruntime.SessionOptions()
            onnxruntime.set_default_logger_severity(3)
            session = onnxruntime.InferenceSession(onnx_model_path,sess_options=session_options,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
            support_att_mask=True
        else:
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=lm_model_file, is_decoder=True,
                                                         torchscript=True) \
                .to(device)
            model.eval()
            if "attention_mask" in inspect.getfullargspec(model.forward).args:
                support_att_mask = True
            else:
                support_att_mask = False

    model_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=lm_model_file,use_fast=True)
    max_seq_length = model_tokenizer.max_len_single_sentence
    if model_tokenizer.pad_token_id is not None if hasattr(model_tokenizer, "pad_token_id") else False:
        pad_id = model_tokenizer.pad_token_id
    elif hasattr(model_tokenizer, "eos_token_id"):
        pad_id = model_tokenizer.eos_token_id
    else:
        pad_id = 0
    vocab= model_tokenizer.get_vocab()
    with open(working_directory+'/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    input_ids, input_mask, actual_token_size = encode\
        (sentence,tokenizer=model_tokenizer,max_seq_length=max_seq_length,pad_id=pad_id)
    if torch.cuda.is_available() and device != "cpu":
        input_ids, input_mask = input_ids.to(device), input_mask.to(device)
    if not (lm_model_file == "gpt2" or lm_model_file == "distilgpt2") or onnx_runtime:
        input_ids, input_mask = input_ids.unsqueeze(0), input_mask.unsqueeze(0)
    print(input_ids.shape[-1],"tokens will be processed")

    if hf_inference or onnx_runtime:
        batch_size = 1  # current_input_ids.size()[0]
        past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
        past = []
        for _ in range(num_layer):
            past.append(torch.empty(*past_shape).type(torch.float32).to(device))

    stride = int(max_seq_length / 2)
    total_score = []
    for i in range(0, input_ids.shape[-1], stride):
        begin_loc = i
        end_loc = min(i + max_seq_length, input_ids.shape[-1])
        current_actual_token_end_loc = min(end_loc, max(actual_token_size+2 - begin_loc, 0))
        trg_len = end_loc - i  # may be different from stride on last loop
        current_input_ids = input_ids[begin_loc:end_loc].to(device)
        current_att_mask = input_mask[begin_loc:end_loc].to(device)
        target_ids = current_input_ids.clone()
        target_ids[:-trg_len] = label_ignore_id

        if hf_inference or onnx_runtime:
            position_ids = None
            if 1:
                # Deduce position_ids from attention mask
                position_ids = (current_att_mask.long().cumsum(-1) - 1)
                position_ids.masked_fill_(current_att_mask==0, 1)
                #position_ids.masked_fill_(position_ids < 0, 0)
            else:
                input_shape = current_input_ids.size()
                if not len(past):
                    past_length = 0
                    past_key_values = [[None] * num_layer]
                else:
                    past_length = past_key_values[0][0].size(-2)
                if position_ids is None:
                    position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long,
                                                device=device)
                    position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        with torch.no_grad():
            if trace_model:
                ts_model=export_torchscript(model=model,input=(current_input_ids, current_att_mask),
                                         quantize=True,device=device)
                torch.jit.save(ts_model, working_directory+"/traced_" + lm_model_file + "_"+device+".pt")
                trace_model = False

            if warmup:
                if hf_inference:
                    model(input_ids=current_input_ids,
                                        attention_mask=current_att_mask if support_att_mask else None,
                                        labels=target_ids,position_ids=position_ids,past_key_values=past)
                else:
                    model(current_input_ids, current_att_mask)

            beginning_of_time = time.time()
            if hf_inference:
                outputs_jit = model(input_ids=current_input_ids,
                                        attention_mask=current_att_mask if support_att_mask else None,
                                        labels=target_ids,position_ids=position_ids,past_key_values=past)
            else:
                if onnx_runtime:
                    my_att_mask=current_att_mask.to(torch.float32)
                    ort_inputs = {'input_ids': np.ascontiguousarray(current_input_ids.cpu().numpy()),
                                  'attention_mask': np.ascontiguousarray(my_att_mask.cpu().numpy()),
                                  'position_ids': np.ascontiguousarray(position_ids.cpu().numpy())
                                  }
                    for i, past_i in enumerate(past):
                        ort_inputs[f'past_{i}'] = np.ascontiguousarray(past_i.cpu().numpy())
                    outputs_jit = session.run(None, ort_inputs)

                else:
                    outputs_jit = model(current_input_ids, current_att_mask)
            time_spent=time.time() - beginning_of_time
            print("timing:", time_spent,". tour")

        past = []
        if hf_inference:
            loss = outputs_jit[0]
            logits = outputs_jit[1]
            past = list(outputs_jit[2])
        else:
            if onnx_runtime:
                del session
                logits=torch.from_numpy(outputs_jit[0].squeeze(0)).to(device)
                loss = process_labels(target_ids,logits)
                for l in range(num_layer):
                    past_i = torch.from_numpy(outputs_jit[l + 1]).clone().detach()
                    past.append(past_i.to(device))
            else:
                loss = process_labels(target_ids, outputs_jit[0])
                logits = outputs_jit[0]

        print(logits.shape)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        print(log_probs.shape)

        if (lm_model_file == "gpt2" or lm_model_file == "distilgpt2") or onnx_runtime:
            my_inputs=input_ids.squeeze(0)
            out_mask=my_inputs[min(max(1, begin_loc),current_actual_token_end_loc):current_actual_token_end_loc]
            target_log_probs = log_probs.gather(-1,
                                                out_mask.unsqueeze(
                                                    -1)).squeeze(-1)
            print(out_mask.shape)
        else:
            out_mask = input_ids[0][min(max(1, begin_loc),current_actual_token_end_loc):current_actual_token_end_loc]
            out_mask = out_mask.unsqueeze(0).unsqueeze(-1)
            target_log_probs = log_probs.gather(-1, out_mask)
            target_log_probs = target_log_probs.squeeze(-1).squeeze(0)

        scores = []

        neg_log_likelihood = loss * trg_len
        scores.append((neg_log_likelihood, neg_log_likelihood.exp(), "neg_log_likelihood h"))

        prod_score = target_log_probs.sum()
        scores.append((prod_score, prod_score.exp(), "prod h"))

        mean_score = target_log_probs.logsumexp(0) - math.log(current_actual_token_end_loc-1) if current_actual_token_end_loc >0 else torch.tensor(0).to(device)
        scores.append((mean_score, mean_score.exp(), "mean h"))

        gmean_score = target_log_probs.mean(0)
        scores.append((gmean_score, gmean_score.exp(), "gmean h"))

        hmean_score = target_log_probs.neg().logsumexp(0).neg() + math.log(current_actual_token_end_loc-1)  if current_actual_token_end_loc >0 else torch.tensor(0).to(device)
        scores.append((hmean_score, hmean_score.exp(), "hmean h"))

        simple_loss_perp = loss
        scores.append((simple_loss_perp, simple_loss_perp.exp(), "loss l"))

        sent_probability = math.exp(-1.0 * simple_loss_perp * (current_actual_token_end_loc-1))  if current_actual_token_end_loc >0 else 0
        sent_probability = float(sent_probability)
        scores.append((torch.tensor(sent_probability), torch.tensor(sent_probability, dtype=float), "sentence prob h"))

        total_score.append(scores)

        if end_loc==input_ids.shape[-1]:
            break

    results = []
    for i in range(7):
        meas = (torch.stack([tour[i][0] for tour in total_score]).sum() / (actual_token_size+1)).item()
        exp = (torch.stack([tour[i][1] for tour in total_score]).sum() / (actual_token_size+1)).item()
        name = total_score[0][i][2]
        entry = (meas, exp, name)
        results.append(entry)

    for result in results:
        print(result)
    print("--------------------------------------------------")

text1="Click on the eye in the icon tray to pick your product of interest or say echelon-connect bike or smart rower. Smart rower."
text2="Click on the eye in the icon tray to pick your product of interest or say echelon-connect bike or smart rower. Smartt roher."
models = [
#    "gpt2",
#    "gpt2-medium",
#    "gpt2-large",
#    "gpt2-xl",
    "distilgpt2",
#    "EleutherAI/gpt-neo-1.3B",
#    "EleutherAI/gpt-neo-125M"
    # See other possible GPT-2 variants (who likes gpt2 style decoder) at https://huggingface.co/models?filter=gpt2
]
for model_name in models:
    if True:
        print(text1)
        nemo_gpt2_test(sentence=text1, lm_model_file=model_name,onnx_runtime=True, trace_model=False,hf_inference=False)
        nemo_gpt2_test(sentence=text1, lm_model_file=model_name,onnx_runtime=False, trace_model=False,hf_inference=True)
        print(text2)
        nemo_gpt2_test(sentence=text2, lm_model_file=model_name,onnx_runtime=True, trace_model=False,hf_inference=False)
        nemo_gpt2_test(sentence=text2, lm_model_file=model_name,onnx_runtime=False, trace_model=False,hf_inference=True)
    print("----------------")


GPT2Config={
  "_name_or_path": "distilgpt2",
  "_num_labels": 1,
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "gradient_checkpointing": False,
  "id2label": {
    "0": "LABEL_0"
  },
  "initializer_range": 0.02,
  "is_decoder": True,
  "label2id": {
    "LABEL_0": 0
  },
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": None,
  "n_layer": 6,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "scale_attn_weights": True,
  "summary_activation": None,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": True,
  "summary_type": "cls_index",
  "summary_use_proj": True,
  "task_specific_params": {
    "text-generation": {
      "do_sample": True,
      "max_length": 50
    }
  },
  "torchscript": True,
  "transformers_version": "4.10.0",
  "use_cache": True,
  "vocab_size": 50257
}
