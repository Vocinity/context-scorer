from typing import Dict, List, Optional
import re
from argparse import Namespace

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.models.transformer import TransformerModel, base_architecture
from fairseq import quantization_utils
from fairseq.data import encoders
from fairseq.models.transformer import EncoderOut
from fairseq import checkpoint_utils
from fairseq.data import Dictionary
from fairseq import search
from fairseq.models import register_model_architecture


@register_model_architecture("transformer", "transformer_g2p")  # type: ignore
def transformer_g2p(args) -> None:  # type: ignore
	args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
	args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
	args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
	args.encoder_layers = getattr(args, "encoder_layers", 3)
	args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
	args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
	args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
	args.decoder_layers = getattr(args, "decoder_layers", 3)
	base_architecture(args)


MODEL_ROOT = './20210722.pt'


class SequenceGenerator(nn.Module):

	def build_embedding(self, num_embeddings, embedding_dim, padding_idx):
		m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
		nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
		nn.init.constant_(m.weight[padding_idx], 0)
		return m

	def __init__(
			self,
			beam_size=5,
			max_len_a=1.2,
			max_len_b=10,
			min_len=1,
			normalize_scores=False,
			len_penalty=1.0,
			unk_penalty=0.0,
			temperature=1.0,
			match_source_len=False,
			no_repeat_ngram_size=0,
			search_strategy=None,
			eos=None,
			symbols_to_strip_from_output=None,
			lm_model=None,
			lm_weight=1.0,
	):
		super().__init__()

		state = checkpoint_utils.load_checkpoint_to_cpu(MODEL_ROOT)
		args = state["args"]

		base_architecture(args)
		encoder_embed_tokens = self.build_embedding(40, 256, 1)
		decoder_embed_tokens = self.build_embedding(80, 256, 1)
		encoder = TransformerModel.build_encoder(args, [], encoder_embed_tokens)
		decoder = TransformerModel.build_decoder(args, ["dummy" * 80], decoder_embed_tokens)
		model = TransformerModel(args, encoder, decoder)

		model = quantization_utils.quantize_model_scalar(model, args)
		model.load_state_dict(state["model"], strict=True, args=args)
		self.model = model

		tgt_dict_data = [("AH0", 56692), ("N", 55136), ("S", 45400), ("L", 44917), ("T", 44154),
						 ("R", 41894), ("K", 38737), ("D", 29326), ("IH0", 27164), ("M", 26774),
						 ("Z", 25321), ("ER0", 21572), ("IY0", 19984), ("B", 19343), ("EH1", 18729),
						 ("P", 17987), ("AA1", 15292), ("AE1", 15263), ("IH1", 14199), ("F", 12565),
						 ("G", 12335), ("V", 9718), ("IY1", 9274), ("NG", 8980), ("HH", 8456),
						 ("EY1", 8451), ("W", 8044), ("SH", 7905), ("OW1", 7806), ("OW0", 7407),
						 ("AO1", 7342), ("AH1", 6237), ("AY1", 6224), ("UW1", 6014), ("JH", 5708),
						 ("Y", 4730), ("AA0", 4470), ("CH", 4454), ("ER1", 4120), ("IH2", 3980),
						 ("EH2", 3527), ("EY2", 3090), ("AE2", 3074), ("AA2", 3046), ("AY2", 3028),
						 ("EH0", 2655), ("TH", 2632), ("IY2", 2367), ("OW2", 2225), ("AW1", 2109),
						 ("UW0", 1862), ("AO2", 1707), ("AE0", 1556), ("UH1", 1438), ("AO0", 1381),
						 ("AY0", 1094), ("UW2", 1085), ("AH2", 1037), ("EY0", 871), ("OY1", 848),
						 ("AW2", 566), ("ER2", 551), ("DH", 531), ("ZH", 517), ("UH2", 432),
						 ("AW0", 332), ("UH0", 234), ("OY2", 191), ("OY0", 111), ("madeupword0000", 0),
						 ("madeupword0001", 0), ("madeupword0002", 0), ("madeupword0003", 0),
						 ("madeupword0004", 0), ("madeupword0005", 0), ("madeupword0006", 0)]

		self.tgt_dict = Dictionary()
		for tgt_dict_item in tgt_dict_data:
			word, n = tgt_dict_item
			self.tgt_dict.add_symbol(word=word, n=n)
		self.pad = self.tgt_dict.pad()
		self.unk = self.tgt_dict.unk()
		self.eos = self.tgt_dict.eos() if eos is None else eos
		self.symbols_to_strip_from_output = (
			symbols_to_strip_from_output.union({self.eos})
			if symbols_to_strip_from_output is not None
			else {self.eos}
		)
		self.vocab_size = len(self.tgt_dict)
		self.beam_size = beam_size
		# the max beam size is the dictionary size - 1, since we never select pad
		self.beam_size = min(beam_size, self.vocab_size - 1)
		self.max_len_a = max_len_a
		self.max_len_b = max_len_b
		self.min_len = min_len

		self.normalize_scores = normalize_scores
		self.len_penalty = len_penalty
		self.unk_penalty = unk_penalty
		self.temperature = temperature
		self.match_source_len = match_source_len
		self.no_repeat_ngram_size = no_repeat_ngram_size
		assert temperature > 0, "--temperature must be greater than 0"

		self.search = (
			search.BeamSearch(self.tgt_dict) if search_strategy is None else search_strategy
		)
		# We only need to set src_lengths in LengthConstrainedBeamSearch.
		# As a module attribute, setting it would break in multithread
		# settings when the model is shared.
		self.should_set_src_lengths = (
				hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
		)

		self.model.eval()

		self.lm_model = lm_model
		self.lm_weight = lm_weight
		if self.lm_model is not None:
			self.lm_model.eval()

	def cuda(self):
		self.model.cuda()
		return self

	def forward_decoder(
			self,
			tokens,
			encoder_out: EncoderOut,
			incremental_states: Dict[str, Dict[str, Optional[Tensor]]],
			temperature: float = 1.0,
	):
		decoder_out = self.model.decoder.forward(
			tokens,
			encoder_out=encoder_out,
			incremental_state=incremental_states)

		attn: Optional[Tensor] = None
		decoder_len = len(decoder_out)
		if decoder_len > 1 and decoder_out[1] is not None:
			attn_holder = decoder_out[1]["attn"]
			if isinstance(attn_holder, Tensor):
				attn = attn_holder
			elif attn_holder is not None:
				attn = attn_holder[0]
			if attn is not None:
				attn = attn[:, -1, :]

		decoder_out_tuple = (
			decoder_out[0][:, -1:, :].div_(temperature),
			None if decoder_len <= 1 else decoder_out[1],
		)

		probs = self.model.get_normalized_probs(
			decoder_out_tuple, log_probs=True, sample=None
		)
		probs = probs[:, -1, :]
		return probs, attn

	@torch.no_grad()
	def forward(self, input):
		prefix_tokens: Optional[Tensor] = None
		constraints: Optional[Tensor] = None
		bos_token: Optional[int] = None
		incremental_states = torch.jit.annotate(
			Dict[str, Dict[str, Optional[Tensor]]], torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})

		)
		net_input = {"src_tokens": input, "src_lengths": torch.tensor(input.numel())}

		src_tokens = input
		# length of the source text being the character length except EndOfSentence and pad
		src_lengths = (
			(src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
		)

		# bsz: total number of sentences in beam
		# Note that src_tokens may have more than 2 dimenions (i.e. audio features)
		bsz, src_len = src_tokens.size()[:2]
		beam_size = self.beam_size

		if constraints is not None and not self.search.supports_constraints:
			raise NotImplementedError(
				"Target-side constraints were provided, but search method doesn't support them"
			)

		# Initialize constraints, when active
		self.search.init_constraints(constraints, beam_size)

		max_len: int = -1
		max_decoder_positions = 1024
		if self.match_source_len:
			max_len = src_lengths.max().item()
		else:
			max_len = min(
				int(self.max_len_a * src_len + self.max_len_b),
				# exclude the EOS marker
				max_decoder_positions - 1,
			)
		assert (
				self.min_len <= max_len
		), "min_len cannot be larger than max_len, please adjust these!"
		# compute the encoder output for each beam
		encoder_outs = self.model.encoder.forward_torchscript(net_input)

		# placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
		new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
		new_order = new_order.to(src_tokens.device).long()
		encoder_outs = self.model.encoder.reorder_encoder_out(encoder_outs, new_order)
		# ensure encoder_outs is a List.
		assert encoder_outs is not None

		# initialize buffers
		scores = (
			torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
		)  # +1 for eos; pad is never chosen for scoring
		tokens = (
			torch.zeros(bsz * beam_size, max_len + 2)
				.to(src_tokens)
				.long()
				.fill_(self.pad)
		)  # +2 for eos and pad
		tokens[:, 0] = self.eos if bos_token is None else bos_token
		attn: Optional[Tensor] = None

		# A list that indicates candidates that should be ignored.
		# For example, suppose we're sampling and have already finalized 2/5
		# samples. Then cands_to_ignore would mark 2 positions as being ignored,
		# so that we only finalize the remaining 3 samples.
		cands_to_ignore = (
			torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
		)  # forward and backward-compatible False mask

		# list of completed sentences
		finalized = torch.jit.annotate(
			List[List[Dict[str, Tensor]]],
			[torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
		)  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

		finished = [
			False for i in range(bsz)
		]  # a boolean array indicating if the sentence at the index is finished or not
		num_remaining_sent = bsz  # number of sentences remaining

		# number of candidate hypos per step
		cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

		# offset arrays for converting between different indexing schemes
		bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
		cand_offsets = torch.arange(0, cand_size).type_as(tokens)

		reorder_state: Optional[Tensor] = None
		batch_idxs: Optional[Tensor] = None

		original_batch_idxs: Optional[Tensor] = None
		original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

		for step in range(max_len + 1):  # one extra step for EOS marker
			# reorder decoder internal states based on the prev choice of beams
			# print(f'step: {step}')
			if reorder_state is not None:
				if batch_idxs is not None:
					# update beam indices to take into account removed sentences
					corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
						batch_idxs
					)
					reorder_state.view(-1, beam_size).add_(
						corr.unsqueeze(-1) * beam_size
					)
					original_batch_idxs = original_batch_idxs[batch_idxs]
				self.model.decoder.reorder_incremental_state_scripting(
					incremental_states, new_order
				)
				encoder_outs = self.model.encoder.reorder_encoder_out(
					encoder_outs, reorder_state
				)

			lprobs, avg_attn_scores = self.forward_decoder(
				tokens[:, : step + 1],
				encoder_outs,
				incremental_states,
				self.temperature,
			)

			lprobs[lprobs != lprobs] = torch.tensor(-float("Inf")).to(lprobs)

			lprobs[:, self.pad] = -float("Inf")  # never select pad
			lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

			# handle max length constraint
			if step >= max_len:
				lprobs[:, : self.eos] = -float("Inf")
				lprobs[:, self.eos + 1:] = -float("Inf")

			# handle prefix tokens (possibly with different lengths)
			if (
					prefix_tokens is not None
					and step < prefix_tokens.size(1)
					and step < max_len
			):
				lprobs, tokens, scores = self._prefix_tokens(
					step, lprobs, scores, tokens, prefix_tokens, beam_size
				)
			elif step < self.min_len:
				# minimum length constraint (does not apply if using prefix_tokens)
				lprobs[:, self.eos] = -float("Inf")

			# Record attention scores, only support avg_attn_scores is a Tensor
			if avg_attn_scores is not None:
				if attn is None:
					attn = torch.empty(
						bsz * beam_size, avg_attn_scores.size(1), max_len + 2
					).to(scores)
				attn[:, :, step + 1].copy_(avg_attn_scores)

			scores = scores.type_as(lprobs)
			eos_bbsz_idx = torch.empty(0).to(
				tokens
			)  # indices of hypothesis ending with eos (finished sentences)
			eos_scores = torch.empty(0).to(
				scores
			)  # scores of hypothesis ending with eos (finished sentences)

			if self.should_set_src_lengths:
				self.search.set_src_lengths(src_lengths)

			if self.no_repeat_ngram_size > 0:
				lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

			# Shape: (batch, cand_size)
			cand_scores, cand_indices, cand_beams = self.search.step(
				step,
				lprobs.view(bsz, -1, self.vocab_size),
				scores.view(bsz, beam_size, -1)[:, :, :step],
				tokens[:, : step + 1],
				original_batch_idxs,
			)

			# cand_bbsz_idx contains beam indices for the top candidate
			# hypotheses, with a range of values: [0, bsz*beam_size),
			# and dimensions: [bsz, cand_size]
			cand_bbsz_idx = cand_beams.add(bbsz_offsets)

			# finalize hypotheses that end in eos
			# Shape of eos_mask: (batch size, beam size)
			eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-float("Inf"))
			eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

			# only consider eos when it's among the top beam_size indices
			# Now we know what beam item(s) to finish
			# Shape: 1d list of absolute-numbered
			eos_bbsz_idx = torch.masked_select(
				cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
			)

			finalized_sents: List[int] = []
			if eos_bbsz_idx.numel() > 0:
				eos_scores = torch.masked_select(
					cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
				)

				finalized_sents = self.finalize_hypos(
					step,
					eos_bbsz_idx,
					eos_scores,
					tokens,
					scores,
					finalized,
					finished,
					beam_size,
					attn,
					src_lengths,
					max_len,
				)
				num_remaining_sent -= len(finalized_sents)

			assert num_remaining_sent >= 0
			if num_remaining_sent == 0:
				break
			if self.search.stop_on_max_len and step >= max_len:
				break
			assert step < max_len

			# Remove finalized sentences (ones for which {beam_size}
			# finished hypotheses have been generated) from the batch.
			if len(finalized_sents) > 0:
				new_bsz = bsz - len(finalized_sents)

				# construct batch_idxs which holds indices of batches to keep for the next pass
				batch_mask = torch.ones(
					bsz, dtype=torch.bool, device=cand_indices.device
				)
				batch_mask[finalized_sents] = False
				# TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
				batch_idxs = torch.arange(
					bsz, device=cand_indices.device
				).masked_select(batch_mask)

				# Choose the subset of the hypothesized constraints that will continue
				self.search.prune_sentences(batch_idxs)

				eos_mask = eos_mask[batch_idxs]
				cand_beams = cand_beams[batch_idxs]
				bbsz_offsets.resize_(new_bsz, 1)
				cand_bbsz_idx = cand_beams.add(bbsz_offsets)
				cand_scores = cand_scores[batch_idxs]
				cand_indices = cand_indices[batch_idxs]

				if prefix_tokens is not None:
					prefix_tokens = prefix_tokens[batch_idxs]
				src_lengths = src_lengths[batch_idxs]
				cands_to_ignore = cands_to_ignore[batch_idxs]

				scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				if attn is not None:
					attn = attn.view(bsz, -1)[batch_idxs].view(
						new_bsz * beam_size, attn.size(1), -1
					)
				bsz = new_bsz
			else:
				batch_idxs = None

			# Set active_mask so that values > cand_size indicate eos hypos
			# and values < cand_size indicate candidate active hypos.
			# After, the min values per row are the top candidate active hypos

			# Rewrite the operator since the element wise or is not supported in torchscript.

			eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
			active_mask = torch.add(
				eos_mask.type_as(cand_offsets) * cand_size,
				cand_offsets[: eos_mask.size(1)],
			)

			# get the top beam_size active hypotheses, which are just
			# the hypos with the smallest values in active_mask.
			# {active_hypos} indicates which {beam_size} hypotheses
			# from the list of {2 * beam_size} candidates were
			# selected. Shapes: (batch size, beam size)
			new_cands_to_ignore, active_hypos = torch.topk(
				active_mask, k=beam_size, dim=1, largest=False
			)

			# update cands_to_ignore to ignore any finalized hypos.
			cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
			# Make sure there is at least one active item for each sentence in the batch.
			assert (~cands_to_ignore).any(dim=1).all()

			# update cands_to_ignore to ignore any finalized hypos

			# {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
			# can be selected more than once).
			active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
			active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

			active_bbsz_idx = active_bbsz_idx.view(-1)
			active_scores = active_scores.view(-1)

			# copy tokens and scores for active hypotheses

			# Set the tokens for each beam (can select the same row more than once)
			tokens[:, : step + 1] = torch.index_select(
				tokens[:, : step + 1], dim=0, index=active_bbsz_idx
			)
			# Select the next token for each of them
			tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
				cand_indices, dim=1, index=active_hypos
			)
			if step > 0:
				scores[:, :step] = torch.index_select(
					scores[:, :step], dim=0, index=active_bbsz_idx
				)
			scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
				cand_scores, dim=1, index=active_hypos
			)

			# Update constraints based on which candidates were selected for the next beam
			self.search.update_constraints(active_hypos)

			# copy attention for active hypotheses
			if attn is not None:
				attn[:, :, : step + 2] = torch.index_select(
					attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
				)

			# reorder incremental state in decoder
			reorder_state = active_bbsz_idx

		# sort by score descending
		for sent in range(len(finalized)):
			scores = torch.tensor(
				[float(elem["score"].item()) for elem in finalized[sent]]
			)
			_, sorted_scores_indices = torch.sort(scores, descending=True)
			finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
			finalized[sent] = torch.jit.annotate(
				List[Dict[str, Tensor]], finalized[sent]
			)
		return finalized

	def _prefix_tokens(
			self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
	):
		"""Handle prefix tokens"""
		prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
		prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
		prefix_mask = prefix_toks.ne(self.pad)
		lprobs[prefix_mask] = torch.tensor(-float("Inf")).to(lprobs)
		lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
			-1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
		)
		# if prefix includes eos, then we should make sure tokens and
		# scores are the same across all beams
		eos_mask = prefix_toks.eq(self.eos)
		if eos_mask.any():
			# validate that the first beam matches the prefix
			first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
						 :, 0, 1: step + 1
						 ]
			eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
			target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
			assert (first_beam == target_prefix).all()

			# copy tokens, scores and lprobs from the first beam to all beams
			tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
			scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
			lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
		return lprobs, tokens, scores

	def replicate_first_beam(self, tensor, mask, beam_size: int):
		tensor = tensor.view(-1, beam_size, tensor.size(-1))
		tensor[mask] = tensor[mask][:, :1, :]
		return tensor.view(-1, tensor.size(-1))

	def finalize_hypos(
			self,
			step: int,
			bbsz_idx,
			eos_scores,
			tokens,
			scores,
			finalized: List[List[Dict[str, Tensor]]],
			finished: List[bool],
			beam_size: int,
			attn: Optional[Tensor],
			src_lengths,
			max_len: int,
	):
		"""Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
		assert bbsz_idx.numel() == eos_scores.numel()

		# clone relevant token and attention tensors.
		# tokens is (batch * beam, max_len). So the index_select
		# gets the newly EOS rows, then selects cols 1..{step + 2}
		tokens_clone = tokens.index_select(0, bbsz_idx)[
					   :, 1: step + 2
					   ]  # skip the first index, which is EOS

		tokens_clone[:, step] = self.eos
		attn_clone = (
			attn.index_select(0, bbsz_idx)[:, :, 1: step + 2]
			if attn is not None
			else None
		)

		# compute scores per token position
		pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
		pos_scores[:, step] = eos_scores
		# convert from cumulative to per-position scores
		pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

		# normalize sentence-level scores
		if self.normalize_scores:
			eos_scores /= (step + 1) ** self.len_penalty

		# cum_unfin records which sentences in the batch are finished.
		# It helps match indexing between (a) the original sentences
		# in the batch and (b) the current, possibly-reduced set of
		# sentences.
		cum_unfin: List[int] = []
		prev = 0
		for f in finished:
			if f:
				prev += 1
			else:
				cum_unfin.append(prev)

		# set() is not supported in script export

		# The keys here are of the form "{sent}_{unfin_idx}", where
		# "unfin_idx" is the index in the current (possibly reduced)
		# list of sentences, and "sent" is the index in the original,
		# unreduced batch
		sents_seen: Dict[str, Optional[Tensor]] = {}

		# For every finished beam item
		for i in range(bbsz_idx.size()[0]):
			idx = bbsz_idx[i]
			score = eos_scores[i]
			# sentence index in the current (possibly reduced) batch
			unfin_idx = idx // beam_size
			# sentence index in the original (unreduced) batch
			sent = unfin_idx + cum_unfin[unfin_idx]
			# print(f"{step} FINISHED {idx} {score} {sent}={unfin_idx} {cum_unfin}")
			# Cannot create dict for key type '(int, int)' in torchscript.
			# The workaround is to cast int to string
			seen = str(sent.item()) + "_" + str(unfin_idx.item())
			if seen not in sents_seen:
				sents_seen[seen] = None

			if self.match_source_len and step > src_lengths[unfin_idx]:
				score = torch.tensor(-float("Inf")).to(score)

			# An input sentence (among those in a batch) is finished when
			# beam_size hypotheses have been collected for it
			if len(finalized[sent]) < beam_size:
				if attn_clone is not None:
					# remove padding tokens from attn scores
					hypo_attn = attn_clone[i]
				else:
					hypo_attn = torch.empty(0)

				finalized[sent].append(
					{
						"tokens": tokens_clone[i],
						"score": score,
						"attention": hypo_attn,  # src_len x tgt_len
						"alignment": torch.empty(0),
						"positional_scores": pos_scores[i],
					}
				)

		newly_finished: List[int] = []

		for seen in sents_seen.keys():
			# check termination conditions for this sentence
			sent: int = int(float(seen.split("_")[0]))
			unfin_idx: int = int(float(seen.split("_")[1]))

			if not finished[sent] and self.is_finished(
					step, unfin_idx, max_len, len(finalized[sent]), beam_size
			):
				finished[sent] = True
				newly_finished.append(unfin_idx)

		return newly_finished

	def is_finished(
			self,
			step: int,
			unfin_idx: int,
			max_len: int,
			finalized_sent_len: int,
			beam_size: int,
	):
		"""
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
		assert finalized_sent_len <= beam_size
		if finalized_sent_len == beam_size or step == max_len:
			return True
		return False

	def calculate_banned_tokens(
			self,
			tokens,
			step: int,
			gen_ngrams: List[Dict[str, List[int]]],
			no_repeat_ngram_size: int,
			bbsz_idx: int,
	):
		tokens_list: List[int] = tokens[
								 bbsz_idx, step + 2 - no_repeat_ngram_size: step + 1
								 ].tolist()
		# before decoding the next token, prevent decoding of ngrams that have already appeared
		ngram_index = ",".join([str(x) for x in tokens_list])
		return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

	def transpose_list(self, l: List[List[int]]):
		# GeneratorExp aren't supported in TS so ignoring the lint
		min_len = min([len(x) for x in l])  # noqa
		l2 = [[row[i] for row in l] for i in range(min_len)]
		return l2

	def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
		# for each beam and batch sentence, generate a list of previous ngrams
		gen_ngrams: List[Dict[str, List[int]]] = [
			torch.jit.annotate(Dict[str, List[int]], {})
			for bbsz_idx in range(bsz * beam_size)
		]
		cpu_tokens = tokens.cpu()
		for bbsz_idx in range(bsz * beam_size):
			gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
			for ngram in self.transpose_list(
					[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
			):
				key = ",".join([str(x) for x in ngram[:-1]])
				gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
					key, torch.jit.annotate(List[int], [])
				) + [ngram[-1]]

		if step + 2 - self.no_repeat_ngram_size >= 0:
			# no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
			banned_tokens = [
				self.calculate_banned_tokens(
					tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
				)
				for bbsz_idx in range(bsz * beam_size)
			]
		else:
			banned_tokens = [
				torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
			]
		for bbsz_idx in range(bsz * beam_size):
			lprobs[bbsz_idx][
				torch.tensor(banned_tokens[bbsz_idx]).long()
			] = torch.tensor(-float("Inf")).to(lprobs)
		return lprobs


def decode_word(model, word, source_dictionary, target_dictionary, tokenizer, with_stress=False, device="cpu") -> str:
	def encode_fn(x: str, tokenizer) -> str:
		if tokenizer is not None:
			x = tokenizer.encode(x)
		return x

	def decode_fn(x: str, tokenizer) -> str:
		if tokenizer is not None:
			x = tokenizer.decode(x)
		return x

	def remove_stress(word: str) -> str:
		letter_pattern = r'[a-zA-Z]+'
		result: List[str] = []
		for elem in word.split():
			match = re.match(letter_pattern, elem)
			elem = match.group(0) if match else elem
			result.append(elem)
		return ' '.join(result)

	def post_process_prediction(
			hypo_tokens,
			src_str,
			alignment,
			align_dict,
			tgt_dict,
			remove_bpe=None,
			extra_symbols_to_ignore=None,
	):
		def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
			from fairseq import tokenizer

			# Tokens are strings here
			hypo_tokens = tokenizer.tokenize_line(hypo_str)
			# TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
			src_tokens = tokenizer.tokenize_line(src_str) + ["<eos>"]
			for i, ht in enumerate(hypo_tokens):
				if ht == unk:
					src_token = src_tokens[alignment[i]]
					# Either take the corresponding value in the aligned dictionary or just copy the original value.
					hypo_tokens[i] = align_dict.get(src_token, src_token)
			return " ".join(hypo_tokens)

		hypo_str = tgt_dict.string(
			hypo_tokens, remove_bpe, extra_symbols_to_ignore=extra_symbols_to_ignore
		)
		if align_dict is not None:
			hypo_str = replace_unk(
				hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
			)
		if align_dict is not None or remove_bpe is not None:
			# Convert back to tokens for evaluating with unk replacement or without BPE
			# Note that the dictionary can be modified inside the method.
			hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
		return hypo_tokens, hypo_str, alignment

	tokens = source_dictionary.encode_line(encode_fn(word, tokenizer), add_if_not_exist=False).long().unsqueeze(0).to(
		device)
	results = []

	translations = model(tokens)
	results.append((tokens[tokens.ne(target_dictionary.pad())], translations[0]))

	result_string = ''
	for src_tokens, hypos in sorted(results, key=lambda x: x[0]):  # type: ignore
		if source_dictionary is not None:
			src_str = source_dictionary.string(src_tokens, "@@")
		for hypo in hypos[: min(len(hypos), 1)]:
			_, hypo_str, _ = post_process_prediction(
				hypo_tokens=hypo["tokens"].int().cpu(),
				src_str=src_str,
				alignment=hypo["alignment"],
				align_dict=None,
				tgt_dict=target_dictionary,
				remove_bpe="@@",
				extra_symbols_to_ignore=[2],
			)
			result_string = decode_fn(hypo_str, tokenizer)
			if not with_stress:
				result_string = remove_stress(result_string)
			break
	return result_string


if __name__ == "__main__":

	src_dict = Dictionary.load("./dict.word.txt")
	tgt_dict = Dictionary.load("./dict.phon.txt")
	tokenizer = encoders.build_tokenizer(Namespace(tokenizer="moses"))

	device = "cpu"
	sequence_generator = SequenceGenerator()

	export_jit = True
	if export_jit:
		sequence_generator = torch.jit.script(sequence_generator)
		sequence_generator.save("transformer_g2p_jit-" + device + ".jit")

	print(decode_word(sequence_generator, "s c r i p t", src_dict, tgt_dict, tokenizer, device=device))
	print(decode_word(sequence_generator, "m o d u l e", src_dict, tgt_dict, tokenizer, device=device))
