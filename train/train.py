import numpy
import numpy as np
import random
import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, \
	get_linear_schedule_with_warmup, \
	AutoModelForCausalLM, AutoTokenizer, AutoConfig, \
	set_seed, \
	default_data_collator, \
	AdamW, get_scheduler
import logging
from datetime import datetime
import math
import csv
from pprint import pprint
import os
import time
from accelerate import Accelerator, DistributedType


class GenerationDataset(Dataset):

	def __init__(self, csv_file_path,
				 tokenizer,
				 is_train=True,
				 eval_split=0.1,
				 constant_block_size=None,
				 special_split_tokens={},
				 phonetic_hacks=None, reverse_q_a_order=False, max_seq_length=1024,
				 label_ignore_id=-100,
				 ):
		if phonetic_hacks is None:
			phonetic_hacks = {}
		self.tokenizer = tokenizer
		pad_token_id = 50256 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

		csv_file = open(csv_file_path)
		csv_reader = csv.reader(csv_file, delimiter=',')

		self.reverse_q_a_order = reverse_q_a_order
		self.phonetic_hacks = phonetic_hacks

		self.dialogs = []
		for idx, row in enumerate(csv_reader):
			if not is_train:
				if idx % int((1 / eval_split)):
					continue
			question = row[0]
			question = self.__preprocess_question(question)
			answer = row[1]
			answer = self.__preprocess_answer(answer)
			self.dialogs.append({"q": question, "a": answer})

		self.samples = []
		if constant_block_size is not None:
			ids = np.array([])
			for dialog in self.dialogs:
				entry = "" if "beginning" not in special_split_tokens else special_split_tokens["beginning"]
				if not reverse_q_a_order:
					entry += dialog["q"] + " " if "fence" not in special_split_tokens else special_split_tokens[
																							   "fence"] + \
																						   dialog[
																							   "a"]
				else:
					entry += dialog["a"] + " " if "fence" not in special_split_tokens else special_split_tokens[
																							   "fence"] + \
																						   dialog["q"]
				entry += tokenizer.special_tokens_map['eos_token'] if "ending" not in special_split_tokens else \
					special_split_tokens["ending"]
				tokens = tokenizer.tokenize(entry)
				ids = np.concatenate((ids, np.array(tokenizer.convert_tokens_to_ids(tokens))))
				if len(ids) >= constant_block_size:
					current_input_ids = torch.from_numpy(ids[:constant_block_size]).to(int)
					current_att_mask = current_input_ids.clone()
					current_att_mask[:] = 1
					elem = {"input_ids": current_input_ids,
							"attention_mask": current_att_mask,
							"labels": current_input_ids
							}
					self.samples.append(elem)
					ids = ids[constant_block_size:]
			while len(ids) >= constant_block_size:
				if len(ids) >= constant_block_size:
					current_input_ids = torch.from_numpy(ids[:constant_block_size]).to(int)
					current_att_mask = current_input_ids.clone()
					current_att_mask[:] = 1
					elem = {"input_ids": current_input_ids,
							"attention_mask": current_att_mask,
							"labels": current_input_ids
							}
					self.samples.append(elem)
					ids = ids[constant_block_size:]
		else:
			for dialog in self.dialogs:
				entry = "" if "beginning" not in special_split_tokens else special_split_tokens["beginning"]
				if not reverse_q_a_order:
					entry += dialog["q"] + " " if "fence" not in special_split_tokens else special_split_tokens["fence"]
					entry += dialog["a"]
				else:
					entry += dialog["a"] + " " if "fence" not in special_split_tokens else special_split_tokens["fence"]
					entry += dialog["q"]

				input_ids, input_mask, actual_token_size = self.__encode \
					(entry, tokenizer=self.tokenizer, max_seq_length=max_seq_length, pad_id=pad_token_id)
				stride = int(max_seq_length / 2)
				for i in range(0, input_ids.shape[-1], stride):
					begin_loc = i
					end_loc = min(i + max_seq_length, input_ids.shape[-1])
					trg_len = end_loc - i  # may be different from stride on last loop
					current_input_ids = input_ids[begin_loc:end_loc]
					current_att_mask = input_mask[begin_loc:end_loc]
					target_ids = current_input_ids.clone()
					target_ids[:-trg_len] = label_ignore_id

					position_ids = (current_att_mask.long().cumsum(-1) - 1)
					position_ids.masked_fill_(current_att_mask == 0, 1)

					elem = {"input_ids": current_input_ids,
							"attention_mask": current_att_mask,
							"labels": target_ids,
							# "position_ids": position_ids
							}
					if not is_train:
						textual = dialog["q"] if not reverse_q_a_order else dialog["a"]
						textual += "" if "fence" not in special_split_tokens else special_split_tokens["fence"]
						q_ids, q_mask, q_length = self.__encode \
							(textual, tokenizer=self.tokenizer,
							 max_seq_length=-1, pad_id=pad_token_id)
						elem["query"] = textual
						elem["response"] = dialog["a"] if not reverse_q_a_order else dialog["q"]
						elem["query_input_id"] = q_ids
						elem["query_att_mask"] = q_mask
					# elem["query_length"] = q_length
					self.samples.append(elem)

					if end_loc == input_ids.shape[-1]:
						break

			if not is_train:
				longest_query = 0
				for i in range(len(self.samples)):
					if "query" in self.samples[i]:
						query_length = len(self.samples[i]["query_input_id"])
						if query_length > longest_query:
							longest_query = query_length
				for i in range(len(self.samples)):
					if "query" in self.samples[i]:
						query_length = len(self.samples[i]["query_input_id"])
						if query_length < longest_query:
							self.samples[i]["query_input_id"] = pad(input=self.samples[i]["query_input_id"],
																	pad=(0, longest_query - query_length),
																	mode='constant', value=pad_token_id)
							self.samples[i]["query_att_mask"] = pad(input=self.samples[i]["query_att_mask"],
																	pad=(0, longest_query - query_length),
																	mode='constant', value=0)

	def __encode(self, text, tokenizer, max_seq_length=1024, pad_id=50256):
		tokens = tokenizer.tokenize(text)
		ids = tokenizer.convert_tokens_to_ids(tokens)
		actual_token_size = len(ids)

		tokens_padded = ids.copy()
		if hasattr(tokenizer, "bos_token_id"):
			tokens_padded = [tokenizer.bos_token_id] + tokens_padded
		if hasattr(tokenizer, "eos_token_id"):
			tokens_padded = tokens_padded + [tokenizer.eos_token_id]
		padded_token_size = len(tokens_padded)

		if max_seq_length == -1:
			max_seq_length = padded_token_size + 1
		full_sequence_size = max_seq_length - (padded_token_size % max_seq_length) + padded_token_size if (
				padded_token_size % max_seq_length) else padded_token_size
		full_sequence = [pad_id] * full_sequence_size
		full_sequence[: padded_token_size] = tokens_padded

		input_mask = np.zeros(full_sequence_size)
		input_mask[: padded_token_size] = 1

		return torch.from_numpy(np.array(full_sequence)), torch.from_numpy(input_mask), actual_token_size

	def __common_preprocessing(self, statement):
		for hack, correct in self.phonetic_hacks.items():
			statement = statement.replace(hack, correct)
		statement = statement.replace("\\n", " ")
		statement = statement.replace("\n", " ")
		statement = statement.replace("   ", " ")
		statement = statement.replace("  ", " ")
		statement = statement.replace(" - ", " ")
		statement = statement.replace(". -- ", " ")
		trailing_chars = [".", "?", "!"]
		for trailing_char in trailing_chars:
			sentences = statement.split(trailing_char)
			if not len(sentences[-1]) or sentences[-1].isspace():
				sentences = sentences[:-1]
			if len(sentences) < 2:
				continue
			capitalized_sentences = []
			for sentence in sentences:
				if not len(sentence):
					continue
				prefix = trailing_char + " " if len(capitalized_sentences) else ""
				sentence = sentence.strip()
				sentence = sentence[0].upper() + sentence[1:]
				capitalized_sentences.append(prefix + sentence)
			statement = ''.join(capitalized_sentences)
		statement = statement.strip()
		return statement[0].upper() + statement[1:]

	def __preprocess_answer(self, answer):
		answer = self.__common_preprocessing(answer)
		return answer

	def __preprocess_question(self, question):
		question = self.__common_preprocessing(question)
		if not question.endswith("?"):
			question += "?"
		return question

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		return self.samples[index]


def train(
		train_dataset, eval_dataset, model, config,
		tokenizer,
		multi_gpu=True,
		device="cpu",
		generate=True,
		batch_size=8, epochs=3, lr=5e-5, weight_decay=0.0,
		warmup_steps=0, max_train_steps=None, output_dir=".",
		output_prefix="context-scorer",
		save_model_on_epoch=True, gradient_accumulation_steps=1
):
	if multi_gpu:
		accelerator = Accelerator(split_batches=True)
		device = accelerator.device
	else:
		model = model.to(device)

	torch.cuda.empty_cache()

	set_seed(42)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
	eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": weight_decay,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

	if multi_gpu:
		model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
			model, optimizer, train_dataloader, eval_dataloader
		)

		if accelerator.distributed_type == DistributedType.TPU:
			model.tie_weights()

	if generate:
		master = True if not multi_gpu else accelerator.is_local_main_process
		if multi_gpu:
			accelerator.wait_for_everyone()
		if master:
			last_eval_sample = eval_dataset[0]
			if "query" in last_eval_sample:
				if multi_gpu:
					initial_model = accelerator.unwrap_model(model)
				else:
					initial_model = model
				initial_model.eval()
				print("Query is: [", last_eval_sample["query"], "]\n")
				greedy_output = initial_model.generate(last_eval_sample["query_input_id"].to(device).unsqueeze(0),
													   max_length=tokenizer.model_max_length,
													   temperature=1.0,
													   top_k=0,
													   top_p=0.9,
													   repetition_penalty=5.0,
													   do_sample=False,
													   num_return_sequences=1,
													   attention_mask=last_eval_sample["query_att_mask"].to(
														   device).unsqueeze(
														   0))
				print("Initially generated: \n[" + tokenizer.decode(greedy_output[0], skip_special_tokens=True) + "]",
					  "\n")

	# Scheduler and math around the number of training steps.
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
	if max_train_steps is None:
		max_train_steps = epochs * num_update_steps_per_epoch

	scheduler = get_scheduler(
		name="linear",
		optimizer=optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=max_train_steps,
	)

	hidden_size = config.n_embd
	num_attention_heads = config.n_head
	num_layer = config.n_layer
	past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
	past = []
	for _ in range(num_layer):
		past.append(torch.zeros(*past_shape).type(torch.float32).to(device))

	completed_steps = 0
	epoch_perplexity = 0
	for epoch in range(epochs):
		torch.cuda.empty_cache()
		master = True if not multi_gpu else accelerator.is_local_main_process
		if master:
			print(f"Training epoch {epoch}")
		model.train()
		loss = torch.Tensor([0])
		for idx, samples in enumerate(train_dataloader):
			beginning_of_time = time.time()
			outputs = model(input_ids=samples["input_ids"].to(device),
							attention_mask=samples["attention_mask"].to(device),
							labels=samples["labels"].to(device),
							# position_ids=samples["position_ids"].to(device) if "position_ids" in samples else None,
							# past_key_values=past
							)
			loss = outputs.loss
			loss = loss / gradient_accumulation_steps
			if multi_gpu:
				accelerator.backward(loss)
			else:
				loss.backward()
			try:
				perplexity = math.exp(loss)
			except OverflowError:
				perplexity = float("inf")

			master = True if not multi_gpu else accelerator.is_local_main_process
			if master:
				time_spent = time.time() - beginning_of_time
				print("Epoch", epoch + 1, "\tIteration ", idx, "/", int(len(train_dataset) / batch_size),
					  "\tLoss:", f'{loss.item():10.8f}',
					  "\tPerplexity",
					  f'{perplexity:10.8f}',
					  "\tLast epoch's perplexity",
					  f'{epoch_perplexity:10.8f}',
					  "\tIteration took:", f'{time_spent:10.8f}', "seconds")

			if (idx % gradient_accumulation_steps == 0) or idx == len(train_dataloader) - 1:
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				completed_steps += 1
				if not multi_gpu:
					model.zero_grad()

			if completed_steps >= max_train_steps:
				break

		model.eval()
		losses = []
		for step, batch in enumerate(eval_dataloader):
			with torch.no_grad():
				outputs = model(input_ids=batch["input_ids"].to(device),
								attention_mask=batch["attention_mask"].to(device),
								labels=batch["labels"].to(device),
								# position_ids=batch["position_ids"].to(device) if "position_ids" in batch else None
								# past_key_values=past
								)

			loss = outputs.loss
			losses.append(accelerator.gather(loss.repeat(batch_size)) if multi_gpu else loss.unsqueeze(0))

		losses = torch.cat(losses)
		losses = losses[: len(eval_dataset)]
		try:
			epoch_perplexity = math.exp(torch.mean(losses))
		except OverflowError:
			epoch_perplexity = float("inf")

		master = True if not multi_gpu else accelerator.is_local_main_process
		if multi_gpu:
			accelerator.wait_for_everyone()
		if master:
			if multi_gpu:
				epoch_model = accelerator.unwrap_model(model)
			else:
				epoch_model = model
			if generate:
				last_eval_sample = eval_dataset[0]
				if "query" in last_eval_sample:
					print("\n", 50 * '# ', "\n")
					print("Generation seed: [", last_eval_sample["query"] + "]")
					greedy_output = epoch_model.generate(last_eval_sample["query_input_id"].to(device).unsqueeze(0),
														 max_length=tokenizer.model_max_length,
														 temperature=1.0,
														 top_k=0,
														 top_p=0.9,
														 repetition_penalty=5.0,
														 do_sample=False,
														 num_return_sequences=1,
														 attention_mask=last_eval_sample["query_att_mask"].to(
															 device).unsqueeze(
															 0)
														 )
					print("Greedy generation output from this epoch was: [",
						  tokenizer.decode(greedy_output[0], skip_special_tokens=True), "]\n")
					print("Had to be something like: ")
					print(last_eval_sample["response"])
			print("\n", 50 * '# ', "\n")
			print("\tEpoch ", epoch, "\tPerplexity:", epoch_perplexity)
		if save_model_on_epoch:
			master = True if not multi_gpu else accelerator.is_local_main_process
			if multi_gpu:
				accelerator.wait_for_everyone()
			if master:
				epoch_model.save_pretrained(os.path.join(output_dir, f"{output_prefix}_{epoch}.pt"),
											save_function=accelerator.save if multi_gpu else torch.save)
				tokenizer.save_pretrained(output_dir)

	if save_model_on_epoch:
		master = True if not multi_gpu else accelerator.is_local_main_process
		if multi_gpu:
			accelerator.wait_for_everyone()
		if master:
			epoch_model.save_pretrained(os.path.join(output_dir, f"{output_prefix}_{epoch}.pt"),
										save_function=accelerator.save if multi_gpu else torch.save)
			tokenizer.save_pretrained(output_dir)
	return model


def main(args):
	my_tokenizer = AutoTokenizer.from_pretrained(args.model_code)
	my_tokenizer.pad_token = my_tokenizer.eos_token
	special_split_tokens ={}# {"beginning": "<|beginning|>", "fence": "<|fence|>", "ending": "<|endoftext|>"}
	#my_tokenizer.add_special_tokens(
	#	{'additional_special_tokens': [special_split_tokens["beginning"], special_split_tokens["fence"]]})

	print("Special tokens:")
	print(my_tokenizer.all_special_tokens)
	print(my_tokenizer.all_special_ids)

	my_model = AutoModelForCausalLM.from_pretrained(args.model_code, use_cache=False)
	my_model.resize_token_embeddings(len(my_tokenizer))
	grad_checkpoint = True
	if grad_checkpoint:
		my_model.gradient_checkpointing_enable()

	config = AutoConfig.from_pretrained(args.model_code)
	pprint(config)

	phonetic_hacks = {"lyve": "live"}
	eval_set = GenerationDataset(csv_file_path="615dcbc2f1223f001a9c3c9c.csv", tokenizer=my_tokenizer, is_train=False,
								 max_seq_length=my_tokenizer.model_max_length,
								 eval_split=args.eval_split,
								 constant_block_size=None,
								 special_split_tokens=special_split_tokens,
								 reverse_q_a_order=args.reverse_columns, phonetic_hacks=phonetic_hacks)

	if not args.eval_only:
		train_set = GenerationDataset(csv_file_path="615dcbc2f1223f001a9c3c9c.csv", tokenizer=my_tokenizer,
									  max_seq_length=my_tokenizer.model_max_length, is_train=True,
									  constant_block_size=None,
									  special_split_tokens=special_split_tokens,
									  reverse_q_a_order=args.reverse_columns, phonetic_hacks=phonetic_hacks)
		# dd/mm/YY H:M:S
		dt_string = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
		train(train_dataset=train_set, tokenizer=my_tokenizer, multi_gpu=args.multi_gpu,
			  device=args.device, generate=args.eval_generation, epochs=args.epochs, config=config,
			  eval_dataset=eval_set, model=my_model, batch_size=args.batch_size, output_dir="./" + dt_string + "/",
			  gradient_accumulation_steps=args.batch_size if grad_checkpoint else 1)
	else:
		my_model.to(args.device)
		my_model.eval()
		eval_dataloader = DataLoader(eval_set, batch_size=1, shuffle=False)
		for step, batch in enumerate(eval_dataloader):
			with torch.no_grad():
				print("\n", step, 50 * '# ', "\n")
				print("Seed is: ", batch["query"])
				greedy_output = my_model.generate(batch["query_input_id"].to(args.device),
												  max_length=my_tokenizer.model_max_length,
												  temperature=1,
												  top_k=0,
												  top_p=0.9,
												  repetition_penalty=5.0,
												  do_sample=False,
												  num_return_sequences=1,
												  attention_mask=batch["query_att_mask"].to(args.device))
				print("Generated: [" + my_tokenizer.decode(greedy_output[0], skip_special_tokens=True) + "]")
				print("Had to be something like:")
				print(batch["response"])


import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		"Have a nice day")
	parser.add_argument("-mgpu", "--multi-gpu", action="store_true", required=False,
						help="Enable HF Accelerate utils")
	parser.add_argument("-dev", "--device", type=str, required=False, default="cuda:0",
						help="Device")
	parser.add_argument("-m", "--model-code", type=str, required=False, default="gpt2",
						help="HF model card name or path to pretrained file")
	parser.add_argument("-b", "--batch-size", type=int, required=False, default=8,
						help="Batch size")
	parser.add_argument("-epo", "--epochs", type=int, required=False, default=100,
						help="Epochs")
	parser.add_argument("-csv", "--csv-path", type=str, required=False, default="615dcbc2f1223f001a9c3c9c.csv",
						help="Dataset path")
	parser.add_argument("-evs", "--eval-split", type=float, required=False, default=0.05,
						help="Eval split")
	parser.add_argument("-eva", "--eval-only", action="store_true", required=False,
						help="Evaluation")
	parser.add_argument("-gva", "--eval-generation", action="store_true", required=False,
						help="Evaluation")
	parser.add_argument("-rc", "--reverse-columns", action="store_true", required=False,
						help="Reverse csv columns")
	args = parser.parse_args()

	main(args)
