import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoModelForCausalLM, set_seed
import logging
import math
import csv
from pprint import pprint
import os
import time
from accelerate import Accelerator


class GenerationDataset(Dataset):

    def __init__(self, csv_file_path,
                 tokenizer,
                 is_train=True,
                 eval_split=0.1,
                 phonetic_hacks=None, reverse_q_a_order=False, max_seq_length=1024,
                 label_ignore_id=-100,
                 hidden_size=768,  # n_embd
                 num_attention_heads=12,  # n_head
                 num_layer=6  # n_layer (hidden layers)
                 ):
        if phonetic_hacks is None:
            phonetic_hacks = {}
        self.tokenizer = tokenizer

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
        for dialog in self.dialogs:
            entry = dialog["q"] + " " + dialog["a"] if not reverse_q_a_order else dialog["a"] + " " + dialog["q"]

            input_ids, input_mask, actual_token_size = self.__encode \
                (entry, tokenizer=self.tokenizer, max_seq_length=max_seq_length)

            stride = int(max_seq_length / 2)
            batch_size = 1  # current_input_ids.size()[0]
            past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
            past = []
            for _ in range(num_layer):
                past.append(torch.empty(*past_shape).type(torch.float32))
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
                        "labels": target_ids, "position_ids": position_ids
                        # , "past_key_values": past
                        }
                if not is_train:
                    q_ids, q_mask, q_length = self.__encode \
                        (dialog["q"] if not reverse_q_a_order else dialog["a"], tokenizer=self.tokenizer, max_seq_length=512)
                    elem["query"] = dialog["q"] if not reverse_q_a_order else dialog["a"]
                    elem["response"] = dialog["a"] if not reverse_q_a_order else dialog["q"]
                    elem["query_input_id"] = q_ids
                    elem["query_att_mask"] = q_mask
                    elem["query_length"] = q_length
                self.samples.append(elem)

                if end_loc == input_ids.shape[-1]:
                    break

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
        train_dataset, eval_dataset, model,
        tokenizer,
        batch_size=8, epochs=100, lr=5e-5,weight_decay=0.0,
        warmup_steps=0, output_dir=".",
        output_prefix="context-scorer",
        save_model_on_epoch=True, gradient_accumulation_steps=1
):
    set_seed(42)

    accelerator = Accelerator(split_batches=True)

    torch.cuda.empty_cache()

    # device = torch.device(device)
    device = accelerator.device
    model = model.to(device)

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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    if accelerator.is_local_main_process:
        initial_model = accelerator.unwrap_model(model)
        initial_model.eval()
        last_eval_sample = eval_dataset[0]
        print("Query is: [",last_eval_sample["query"],"]\n")
        greedy_output = initial_model.generate(last_eval_sample["query_input_id"].to(device).unsqueeze(0),
                                          max_length=1024,
                                          temperature=1.0,
                                          top_k=0,
                                          top_p=0.9,
                                          repetition_penalty=5.0,
                                          do_sample=False,
                                          num_return_sequences=1,
                                          attention_mask=last_eval_sample["query_att_mask"].to(device).unsqueeze(0))
        print("Initially generated: \n[" + tokenizer.decode(greedy_output[0], skip_special_tokens=True) + "]", "\n")


    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    epoch_loss = torch.Tensor([0])
    for epoch in range(epochs):
        if accelerator.is_local_main_process:
            print(f"Training epoch {epoch}")
            younger_model = accelerator.unwrap_model(model)
        model.train()
        loss = torch.Tensor([0])
        for idx, samples in enumerate(train_dataloader):
            beginning_of_time = time.time()

            outputs = model(input_ids=samples["input_ids"],
                            attention_mask=samples["attention_mask"],
                            labels=samples["labels"], position_ids=samples["position_ids"]
                            # ,past_key_values=sample["past_key_values"]
                            )

            loss = outputs[0]
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if (idx % gradient_accumulation_steps == 0) or idx == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # model.zero_grad()

            if accelerator.is_local_main_process:
                time_spent = time.time() - beginning_of_time
                print("Epoch", epoch, "\tIteration ", idx, "/", int(len(train_dataset) / (batch_size)),
                      "\tLoss:", f'{loss.item():10.8f}',
                      "\tLast epoch loss",
                      epoch_loss.item(), "\tIteration took:", f'{time_spent:10.8f}', "seconds")

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"], position_ids=batch["position_ids"]
                                # ,past_key_values=sample["past_key_values"]
                                )

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        if accelerator.is_local_main_process:
            last_eval_sample = eval_dataset[0]
            print("\n", 50 * '# ', "\n")
            print("Generation seed: [",last_eval_sample["query"] + "]")
            younger_model.eval()
            greedy_output = younger_model.generate(last_eval_sample["query_input_id"].to(device).unsqueeze(0),
                                                   max_length=1024,
                                                   temperature=1.0,
                                                   top_k=0,
                                                   top_p=0.9,
                                                   repetition_penalty=5.0,
                                                   do_sample=False,
                                                   num_return_sequences=1,
                                                   attention_mask=last_eval_sample["query_att_mask"].to(device).unsqueeze(0))
            print("Greedy generation output from previous epoch was: [", tokenizer.decode(greedy_output[0], skip_special_tokens=True),"]\n")
            older_model = accelerator.unwrap_model(model)
            greedy_output = older_model.generate(last_eval_sample["query_input_id"].to(device).unsqueeze(0),
                                                 max_length=1024,
                                                 temperature=1.0,
                                                 top_k=0,
                                                 top_p=0.9,
                                                 repetition_penalty=5.0,
                                                 do_sample=False,
                                                 num_return_sequences=1,
                                                 attention_mask=last_eval_sample["query_att_mask"].to(device).unsqueeze(0)
                                                 )
            print("Greedy generation output from this epoch was: [",
                  tokenizer.decode(greedy_output[0], skip_special_tokens=True), "]\n")
            print("Had to be something like: ")
            print(last_eval_sample["response"])
            print("\n", 50 * '# ', "\n")
            print("\tEpoch ", epoch, " loss:", loss.item(), " perplexity:", perplexity)
        epoch_loss = loss
        if save_model_on_epoch:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                older_model.save_pretrained(os.path.join(output_dir, f"{output_prefix}_{epoch}.pt"),
                                            save_function=accelerator.save)
                # if accelerator.is_main_process:
                #    tokenizer.save_pretrained(output_dir)
    return model


def main(model_to_be_validated="/opt/cloud/projects/vocinity/context-scorer/train/context-scorer__1.pt"):
    phonetic_hacks = {"lyve": "live"}
    model_code = "gpt2"

    my_tokenizer = GPT2Tokenizer.from_pretrained(model_code)
    my_model = AutoModelForCausalLM.from_pretrained(model_code if model_to_be_validated is None else model_to_be_validated,use_cache=False)
    my_model.resize_token_embeddings(len(my_tokenizer))
    my_model.gradient_checkpointing_enable()

    eval_set = GenerationDataset(csv_file_path="615dcbc2f1223f001a9c3c9c.csv", tokenizer=my_tokenizer, is_train=False,
                                 max_seq_length=1024, eval_split=0.05,
                                 reverse_q_a_order=False, phonetic_hacks=phonetic_hacks)

    if model_to_be_validated is None:
        train_set = GenerationDataset(csv_file_path="615dcbc2f1223f001a9c3c9c.csv", tokenizer=my_tokenizer,
                                      max_seq_length=1024, is_train=True,
                                      reverse_q_a_order=False, phonetic_hacks=phonetic_hacks)
        train(train_dataset=train_set, tokenizer=my_tokenizer, eval_dataset=eval_set, model=my_model, batch_size=10,
              gradient_accumulation_steps=10)
    else:
        my_model.to("cpu")
        my_model.eval()
        eval_dataloader = DataLoader(eval_set, batch_size=1, shuffle=False)
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                print("\n",step, 50 * '# ', "\n")
                print("Seed is: ",batch["query"])
                greedy_output = my_model.generate(batch["query_input_id"].to("cpu"),
                                                  max_length=1024,
                                                  temperature=1,
                                                  top_k=0,
                                                  top_p=0.9,
                                                  repetition_penalty=5.0,
                                                  do_sample=False,
                                                  num_return_sequences=1,
                                                  attention_mask=batch["query_att_mask"].to("cpu"))
                print("Generated: [" + my_tokenizer.decode(greedy_output[0], skip_special_tokens=True) + "]")
                print("Had to be something like:")
                print(batch["response"])


if __name__ == "__main__":
    main(None)
