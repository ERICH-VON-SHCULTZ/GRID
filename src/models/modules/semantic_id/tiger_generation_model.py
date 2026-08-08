import collections
import logging
from typing import Any, List, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torchmetrics.aggregation import BaseAggregator
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.models.t5.modeling_t5 import T5Config, T5LayerNorm

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.components.loss_functions import FocalLoss
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.components.network_blocks.mlp import MLP
from src.models.modules.huggingface.transformer_base_module import TransformerBaseModule
from src.utils.codebook_embedding_init import build_codebook_init_embedding
from src.utils.utils import (
    delete_module,
    find_module_shape,
    get_parent_module_and_attr,
    reset_parameters,
)


class SemanticIDGenerativeRecommender(TransformerBaseModule):
    """
    This is a base class for the generative recommender model.
    It is used to generate the semantic ID for the given input.
    It does not contain any specific implementation for the encoder or decoder.
    The encoder and decoder are defined in the subclasses.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        embedding_dim: int,
        should_check_prefix: bool,
        top_k_for_generation: int,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDGenerativeRecommender module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        num_embeddings_per_hierarchy (int): the number of embeddings per hierarchy.
        embedding_dim (int): the dimension of the embeddings.
        top_k_for_generation (int): the number of top-k candidates for generation.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """
        super().__init__(**kwargs)

        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.embedding_dim = embedding_dim
        self.num_hierarchies = num_hierarchies
        self.should_check_prefix = should_check_prefix
        if codebooks != None:
            self.codebooks = codebooks.t()
            assert (
                self.codebooks.size(1) == num_hierarchies
            ), "codebooks should be of shape (-1, num_hierarchies)"
        else:
            logging.warning(
                "Not using pre-cached codebooks, \
            please make sure that \n \
                            1) dataset is properly pre-processed \n \
                            2) num_hierarchies and  num_embeddings_per_hierarchy are proerly set\
            "
            )

        self.top_k_for_generation = top_k_for_generation

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject a separator token into the ID embeddings and attention mask.

        Parameters:
        id_embeddings (torch.Tensor): The ID embeddings of shape (batch_size, seq_len, emb_dim).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
        sep_token (torch.Tensor): The separator token of shape (1, emb_dim).
        num_hierarchies (int): The number of hierarchies in the codebooks.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified ID embeddings and attention mask.
        id_embeddings: The ID embeddings with the separator token injected of shape (batch_size, seq_len + num_items, emb_dim).
        attention_mask: The attention mask with the separator token injected of shape (batch_size, seq_len + num_items).

        An intuitive example of the input and output:
        input:
        id_embeddings: [[1, 2, 3, 4], [5, 6, 7, 8]]
        attention_mask: [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        output:
        id_embeddings: [[1, 2, 3, 4, sep_token], [5, 6, 7, 8, sep_token]]
        attention_mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
        """
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count_per_sequence = seq_len // num_hierarchies

        reshaped_id_embeddings = id_embeddings.view(
            batch_size, item_count_per_sequence, num_hierarchies, -1
        )
        reshaped_attention_mask = attention_mask.view(
            batch_size, item_count_per_sequence, num_hierarchies
        )
        reshaped_sep_token_for_concat = (
            sep_token.unsqueeze(0)
            .expand(batch_size, item_count_per_sequence, -1)
            .unsqueeze(-2)
        )
        id_embeddings = torch.cat(
            [reshaped_id_embeddings, reshaped_sep_token_for_concat], dim=-2
        )
        attention_mask = torch.cat(
            [reshaped_attention_mask, reshaped_attention_mask[:, :, [-1]]],
            dim=-1,
        )
        id_embeddings = id_embeddings.reshape(batch_size, -1, emb_dim)
        attention_mask = attention_mask.reshape(batch_size, -1)
        return id_embeddings, attention_mask

    def _spawn_embedding_tables(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> torch.nn.Embedding:
        """
        Spawn an embedding table with the given number of embeddings and embedding dimension.

        Parameters:
        num_embeddings (int): the number of embeddings in the table.
        embedding_dim (int): the dimension of the embeddings.
        """
        table = torch.nn.Embedding(
            num_embeddings=num_embeddings,  # type: ignore
            embedding_dim=embedding_dim,  # type: ignore
        )
        return table

    def _is_kv_cache_valid(
        self, kv_cache: Union[Tuple, DynamicCache, EncoderDecoderCache]
    ) -> bool:

        if isinstance(kv_cache, (EncoderDecoderCache, DynamicCache)):
            return len(kv_cache) > 0
        elif isinstance(kv_cache, Tuple):
            return True
        else:
            return False

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Adds repeating offsets to each element in each row of input_sids.
        we use a single embedding table for multiple code books.
        for example if each codebook has 300 embeddings and we have 3 codebooks,
        the input sequence will be transformed from [0, 1, 2] -> to [0, 301, 602]

        Parameters:
            input_sids (torch.Tensor): A 2D PyTorch tensor.
            codebook_size (int): The number of elements in the codebook.
            num_hierarchies (int): The number of hierarchy levels.
        """

        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")

        num_rows, num_cols = input_sids.shape
        offsets = (
            torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        )

        # Calculate how many times the full offset pattern needs to repeat
        num_repeats = (
            num_cols + num_hierarchies - 1
        ) // num_hierarchies  # Integer division to handle cases where num_cols is not a multiple of num_hierarchies

        # Repeat the offsets and slice to match the number of columns
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]

        # Add the repeated offsets to each row using broadcasting
        input_sids_with_offsets = input_sids + repeated_offsets
        if attention_mask is not None:
            input_sids_with_offsets = input_sids_with_offsets * attention_mask
        return input_sids_with_offsets

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        """
        Checks if a given prefix is a valid prefix of the codebooks.

        Args:
            prefix: A tensor of shape [batch_size, hierarchy_level].
            batch_size: The size of the batch to process.

        Returns:
            A boolean tensor of shape [batch_size] indicating the validity of each prefix.
        """
        # TODO (clark): this is a temporary solution, we should use a more efficient way to do this
        # like pre-sorting the codebook and implementing a tree strcture

        current_hierarchy = prefix.shape[1]
        num_prefixes = prefix.shape[0]
        results = []

        # Ensure codebooks are on the correct device.  Do this *once* outside the loop.
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)

        # Trim the codebooks to the relevant hierarchy *once* outside the loop.
        trimmed_codebooks = self.codebooks[:, :current_hierarchy]

        for i in range(0, num_prefixes, batch_size):
            # Get the current batch of prefixes.
            batch_prefix = prefix[
                i : i + batch_size
            ]  # Shape: [batch_size, hierarchy_level]

            # Perform the comparison.  Broadcasting is now limited by batch_size.
            # trimmed_codebooks shape: [C, H] -> unsqueezed [C, 1, H]
            # batch_prefix shape   : [b, H] -> unsqueezed [1, b, H]
            # comparison result    : [C, b, H]
            comparison = trimmed_codebooks.unsqueeze(1) == batch_prefix.unsqueeze(0)

            # Reduce along the hierarchy dimension (H). Shape: [C, b]
            all_match = comparison.all(dim=2)

            # Reduce along the codebook dimension (C).  Shape: [b]
            any_match = all_match.any(dim=0)

            # Append the results for this batch.
            results.append(any_match)

        # Concatenate the results from all batches.
        return torch.cat(results)

    def _beam_search_one_step(
        self,
        candidate_logits: torch.Tensor,
        generated_ids: Union[torch.Tensor, None],
        marginal_log_prob: Union[torch.Tensor, None],
        past_key_values: Union[EncoderDecoderCache, None],
        hierarchy: int,
        batch_size: int,
        num_emb_override: Optional[int] = None,
    ):
        """
        Perform one step of beam search.

        Args:
            candidate_logits: The logits for the next token.
            generated_ids: The generated IDs so far.
            marginal_log_prob: The marginal log probabilities.
            past_key_values: The cache for past key values.
            hierarchy: The current hierarchy level.
            batch_size: The size of the batch.

        Returns:
            The updated generated IDs and the marginal probabilities.
        """

        num_emb = num_emb_override if num_emb_override is not None else self.num_embeddings_per_hierarchy

        # pruning the beams that cannot be mapped to a valid item
        if self.should_check_prefix:
            if generated_ids is None:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.arange(
                        num_emb,
                        device=candidate_logits.device,
                    ).unsqueeze(1)
                )
                candidate_logits[:, ~valid_prefix_mask] = float("-inf")
            else:
                # we prune all beams with prefixes that cannot be mapped to a valid item
                valid_prefix_mask = self._check_valid_prefix(
                    torch.cat(
                        [
                            generated_ids.reshape(-1, hierarchy).repeat_interleave(
                                num_emb, dim=0
                            ),
                            torch.arange(
                                num_emb,
                                device=candidate_logits.device,
                            )
                            .repeat(self.top_k_for_generation * batch_size)
                            .unsqueeze(1),
                        ],
                        dim=1,
                    )
                ).reshape(-1, num_emb)
            candidate_logits[~valid_prefix_mask] = float("-inf")

        candidate_logits = torch.nn.functional.softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if generated_ids is None:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )
            generated_ids = indices_topk.unsqueeze(-1)
            # we need to overwrite the cache because we expanded the beam width from bsz to bsz * beam_width
            # real KV cache starts from the first hierarchy rather than 0-th
            # this is because in 0th hierarchy, self-attention doesn't have cache.
            # and kv cache in huggingface has poor support for this corner case
            past_key_values = EncoderDecoderCache(
                self_attention_cache=DynamicCache(),
                cross_attention_cache=DynamicCache(),
            )
            replace_indices = None
        else:
            # we have beams, generating more beams from the existing beams
            proba, indices = (
                proba[:, : num_emb],
                indices[:, : num_emb],
            )
            proba, indices = proba.reshape(
                -1, self.top_k_for_generation * num_emb
            ), indices.reshape(
                -1, self.top_k_for_generation * num_emb
            )
            # calculating the marginal probability
            proba = torch.mul(
                marginal_log_prob.repeat_interleave(
                    num_emb, dim=-1
                ),
                proba,
            )
            topk_results = torch.topk(
                torch.nan_to_num(proba, nan=-1), k=self.top_k_for_generation, dim=-1
            )
            proba_topk, indices_topk = topk_results.values, topk_results.indices
            # getting indices of winning beams in the original beams
            replace_indices = (
                (indices_topk // num_emb)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1)
                * self.top_k_for_generation
            ).flatten()
            # accordingly update kv cache given the winning beams
            if past_key_values != None:
                past_key_values.reorder_cache(replace_indices)

            indices_topk = torch.gather(indices, 1, indices_topk)

        if replace_indices != None:
            generated_ids = torch.cat(
                [
                    generated_ids.reshape(-1, hierarchy)[replace_indices].reshape(
                        -1, self.top_k_for_generation, hierarchy
                    ),
                    indices_topk.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            generated_ids = indices_topk.unsqueeze(-1)

        return generated_ids, proba_topk, past_key_values

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
    ):
        """Perform a single evaluation step on a batch of data from the validation or test set.
        The method will update the metrics and the loss that is passed.
        """
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        _, loss = self.model_step(model_input=model_input, label_data=label_data)

        generated_ids, marginal_probs = self.generate(
            attention_mask=model_input.mask,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            # TODO: (lneves) hardcoded for now, will need to change for multiple features
            labels=list(label_data.labels.values())[0].to(marginal_probs.device),
        )

        loss_to_aggregate(loss)

    def _make_deterministic(self, is_training: bool):
        """
        Make the model deterministic by turning off some flags.
        This is needed as the default functions in lightning such as
        on_validation_start on_predict_start cannnot properly set the flags
        for the encoder and decoder.
        (TODO) clark: in the future we can revisit this and make it more generic

        Args:
            is_training (bool): Whether the model is in training mode or not.
        """
        if is_training:
            if self.decoder != None:
                self.decoder.decoder.is_training = True
                self.decoder.decoder.train()
            if self.encoder != None:
                self.encoder.encoder.is_training = True
                self.encoder.encoder.train()
        else:
            if self.decoder != None:
                self.decoder.decoder.is_training = False
                self.decoder.decoder.eval()
            if self.encoder != None:
                self.encoder.encoder.is_training = False
                self.encoder.encoder.eval()

    def on_predict_start(self):
        super().on_predict_start()
        self._make_deterministic(is_training=False)

    def on_predict_end(self):
        super().on_predict_end()
        self._make_deterministic(is_training=True)

    def on_validation_start(self):
        super().on_validation_start()
        self._make_deterministic(is_training=False)

    def on_validation_end(self):
        super().on_validation_end()
        self._make_deterministic(is_training=True)

    def on_test_start(self):
        super().on_test_start()
        self._make_deterministic(is_training=False)

    def on_test_end(self):
        super().on_test_end()
        self._make_deterministic(is_training=True)

    def on_train_start(self):
        super().on_train_start()
        self._make_deterministic(is_training=True)


class SemanticIDEncoderDecoder(SemanticIDGenerativeRecommender):
    """
    This is an in-house implementation of the encoder-decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    We added some additional features and modifications to the original architecture.
    (e.g., constrained beam search, separation tokens, etc)
    """

    def __init__(
        self,
        top_k_for_generation: int = 10,
        codebooks: torch.Tensor = None,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        mlp_layers: Optional[int] = None,
        should_check_prefix: bool = False,
        should_add_sep_token: bool = True,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "semantic_ids",
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDEncoderDecoder module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings_per_hierarchy).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        top_k_for_generation (int): the number of top-k candidates for generation.
        num_user_bins (Optional[int]): the number of bins for user in the dataset (this number equals to the number of rows in the embedding table ).
        mlp_layers (Optional[int]): the number of mlp layers in the encoder and decoder.
        embedding_dim (Optional[int]): the dimension of the embeddings.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """

        if num_hierarchies is None or num_embeddings_per_hierarchy is None:
            num_hierarchies, num_embeddings_per_hierarchy = (
                codebooks.shape[0],
                codebooks.max().item() + 1,
            )
        if embedding_dim is None:
            embedding_dim = (
                kwargs["huggingface_model"]
                .encoder.block[0]
                .layer[0]
                .SelfAttention.q.in_features
            )

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        self.encoder = SemanticIDEncoderModule(
            encoder=self.encoder,
        )

        # bos_token used to prompt the decoder to generate the first token
        bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )

        self.decoder = SemanticIDDecoderModule(
            decoder=self.decoder,
            bos_token=bos_token,
            decoder_mlp=torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.embedding_dim,
                        self.num_embeddings_per_hierarchy,
                        bias=False,
                    )
                    for _ in range(self.num_hierarchies)
                ]
            ),
        )

        if mlp_layers is not None:
            # bloating the mlp layers in both encoder and decoder
            # TODO (clark): this currently only works for T5
            for name, module in self.named_modules():
                if isinstance(module, transformers.models.t5.modeling_t5.T5LayerFF):
                    parent_module, attr_name = get_parent_module_and_attr(self, name)
                    setattr(
                        parent_module,
                        attr_name,
                        T5MultiLayerFF(
                            config=self.encoder.encoder.config, num_layers=mlp_layers
                        ),
                    )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )

        # generating user embedding table
        self.user_embedding: torch.nn.Embedding = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        # separation token for the encoder to differentiate between items
        self.sep_token = (
            torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
            if should_add_sep_token
            else None
        )
        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name

    def encoder_forward_pass(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder module.

        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """

        # we shift the IDs here to match the hierarchy structure
        # so that we can use a single embedding table to store the embeddigns for all hierarchies
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds_for_encoder = self.get_embedding_table(table_name="encoder")(
            shifted_sids
        )

        if self.sep_token is not None:
            (
                inputs_embeds_for_encoder,
                attention_mask,
            ) = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds_for_encoder,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        # we enter this loop if we want to use user_id
        if user_id is not None and self.user_embedding is not None:
            # preprocessing function pad user_id with zeros
            # so we only need to take the first column
            user_id = user_id[:, 0]

            # TODO (clark): here we assume remainder hashing, which is different from LSH hashing used in TIGER.
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )

            # prepending the user_id embedding to the input senquence
            inputs_embeds_for_encoder = torch.cat(
                [
                    user_embeds.unsqueeze(1),
                    inputs_embeds_for_encoder,
                ],
                dim=1,
            )
            # prepending 1 to attention mask as we introduce user embedding in the first column
            user_attention_mask = torch.ones(
                attention_mask.size(0), 1, device=attention_mask.device
            )
            attention_mask_for_encoder = torch.cat(
                [
                    user_attention_mask,
                    attention_mask,
                ],
                dim=1,
            )
        else:
            attention_mask_for_encoder = attention_mask

        encoder_output = self.encoder(
            sequence_embedding=inputs_embeds_for_encoder,
            attention_mask=attention_mask_for_encoder,
        )
        return encoder_output, attention_mask_for_encoder

    def decoder_forward_pass(
        self,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # TODO (clark): in the future we should support variable length semantic id
        future_ids: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask_for_encoder: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            encoder_output (Optional[torch.Tensor]): The output from the encoder.
            attention_mask_for_encoder (Optional[torch.Tensor]): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (Optional[DynamicCache]): The cache for past key values.
        """

        # we generated something before and we need to shift the future_ids
        if future_ids is not None:
            shifted_future_sids = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids, device=future_ids.device)
                if attention_mask is None
                else attention_mask,
            )
            inputs_embeds_for_decoder = self.get_embedding_table(table_name="decoder")(
                shifted_future_sids
            )

            # we do not have valid kv cache
            # we need to prepend bos token to the decoder input
            if not self._is_kv_cache_valid(kv_cache=past_key_values):
                inputs_embeds_for_decoder = torch.cat(
                    [
                        self.decoder.bos_token.unsqueeze(0).expand(
                            future_ids.size(0), 1, -1
                        ),
                        inputs_embeds_for_decoder,
                    ],
                    dim=1,
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(future_ids.size(0), 1, device=future_ids.device),
                            attention_mask,
                        ],
                        dim=1,
                    )
            else:
                # we have valid kv cache
                # we only need the last token in the decoder input
                inputs_embeds_for_decoder = inputs_embeds_for_decoder[:, -1:, :]
        # this is the beginning of generation, we start from bos token
        else:
            inputs_embeds_for_decoder = self.decoder.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

        decoder_output = self.decoder(
            sequence_embedding=inputs_embeds_for_decoder,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask_for_encoder,
            encoder_output=encoder_output,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        return decoder_output

    def generate(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """

        # getting encoder output
        # we only need to do this once because we have decoder
        # to do auto-regressive generation
        encoder_output, encoder_attention_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )

        # initilize cached generated ids to None
        generated_ids = None
        marginal_log_prob = None

        # initialize kv cache
        past_key_values = EncoderDecoderCache(
            self_attention_cache=DynamicCache(), cross_attention_cache=DynamicCache()
        )

        for hierarchy in range(self.num_hierarchies):
            if generated_ids is not None:
                # we generated something before
                # we need to reshape the generated ids so that
                # the number of beams equals to batch size * top_k
                squeezed_generated_ids = generated_ids.reshape(-1, hierarchy).to(
                    encoder_output.device
                )  # shape: (batch_size * top_k, hierarchy)

                repeated_encoder_output = encoder_output.repeat_interleave(
                    self.top_k_for_generation, dim=0
                )
                # shape: (batch_size * top_k, seq_len+1, hidden_dim)
                # +1 because we have user_id token

                repeated_encoder_attention_mask = (
                    encoder_attention_mask.repeat_interleave(
                        self.top_k_for_generation, dim=0
                    )
                )  # shape: (batch_size * top_k, seq_len+1)
            else:
                # we haven't generated anything yet!
                # the number of beams currently equals to batch size
                squeezed_generated_ids = None
                repeated_encoder_output = encoder_output
                repeated_encoder_attention_mask = encoder_attention_mask

            # feeding the decoder with the generated ids
            decoder_output, past_key_values = self.decoder_forward_pass(
                future_ids=squeezed_generated_ids,
                encoder_output=repeated_encoder_output,
                attention_mask_for_encoder=repeated_encoder_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            # decoder_output[:, -1, :] is the embedding for the next token
            latest_output_representation = decoder_output[:, -1, :]

            # # calculating the logits for the next token
            candidate_logits = self.decoder.decoder_mlp[hierarchy](
                latest_output_representation
            )  # shape: (batch_size * top_k, num_embeddings in the hierarchy)

            (
                generated_ids,
                marginal_log_prob,
                past_key_values,
            ) = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                past_key_values=past_key_values,
                hierarchy=hierarchy,
                batch_size=input_ids.size(0),
            )

        return generated_ids, marginal_log_prob

    def forward(
        self,
        attention_mask_encoder: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
        future_ids: Optional[torch.Tensor] = None,
        attention_mask_decoder: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.
        Parameters:
            attention_mask_encoder (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            attention_mask_decoder (Optional[torch.Tensor]): The attention mask for the decoder.
        """

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask_encoder,
            input_ids=input_ids,
            user_id=user_id,
        )

        decoder_output = self.decoder_forward_pass(
            future_ids=future_ids,
            attention_mask=attention_mask_decoder,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,  # we are not using cache for training
        )
        return decoder_output

    def get_embedding_table(self, table_name: str, hierarchy: Optional[int] = None):
        """
        Get the embedding table for the given table name and hierarchy.
        Args:
            table_name: The name of the table to get the embedding for.
            hierarchy: The hierarchy level to get the embedding for.
        """
        # here we assume the encoder and decoder share the same embedding table
        # we can have flexible embedding table in the future
        if table_name == "encoder":
            embedding_table = self.item_sid_embedding_table_encoder
        elif table_name == "decoder":
            embedding_table = self.item_sid_embedding_table_encoder

        if hierarchy is not None:
            return embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return embedding_table

    def predict_step(self, batch: SequentialModelInputData):
        generated_sids, _ = self.model_step(batch)
        ids = [
            id.item() if isinstance(id, torch.Tensor) else id
            for id in batch.user_id_list
        ]
        model_output = OneKeyPerPredictionOutput(
            keys=ids,
            predictions=generated_sids,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_value_name,
        )
        return model_output

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the model and calculate the loss if label_data is provided.

        Args:
            model_input: The input data to the model.
            label_data: The label data to the model. Its optional as it is not required for inference.
        """

        # if label_data is None, we are in inference mode and doing free-form generation
        if label_data is None:
            # this is inference stage
            generated_ids, marginal_probs = self.generate(
                attention_mask=model_input.mask,
                **{
                    self.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                },
            )
            return generated_ids, 0  # returning 0 here because we don't have a loss

        fut_ids = None
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            fut_ids = curr_label.reshape(model_input.mask.size(0), -1)
        # here we pass labels in to the forward function
        # because the decoder is causal and we are doing shifted prediction
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        # we prepended a bos token to the decoder input
        # so we need to remove the last token in the output
        model_output = model_output[:, :-1]

        # the label locations is shared for all semantic id hierarchies
        loss = 0
        for hierarchy in range(self.num_hierarchies):

            input = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
            loss += self.loss_function(
                input=input,
                target=fut_ids[:, hierarchy].long(),
            )
        return model_output, loss


class SemanticIDDecoderModule(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        decoder_mlp: Optional[torch.nn.Module] = None,
        bos_token: Optional[torch.nn.Parameter] = None,
    ) -> None:
        """
        Initialize the SemanticIDDecoderModule.

        Parameters:
        decoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        decoder_mlp (torch.nn.Module): the mlp layers used to project the decoder output to the embedding table.
        bos_token (Optional[torch.nn.Parameter]):
            the bos token used to prompt the decoder.
            if None, then this means the decoder is used standalone without an encoder.
        """

        super().__init__()
        # some sanity checks
        if bos_token is not None:
            assert decoder.config.is_decoder == True, "Decoder must be a decoder model"
            assert (
                decoder.config.is_encoder_decoder == False
            ), "Decoder must be a standalone decoder model"

        self.decoder = decoder
        # this bos token is prompt for the decoder
        self.bos_token = bos_token
        self.decoder_mlp = decoder_mlp
        # deleting embedding table in the decoder to save space
        delete_module(self.decoder, "embed_tokens")
        delete_module(self.decoder, "shared")
        reset_parameters(self.decoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache = DynamicCache(),
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            sequence_embedding (torch.Tensor): The input sequence embedding for the decoder.
            encoder_output (torch.Tensor): The output from the encoder.
            encoder_attention_mask (torch.Tensor): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (DynamicCache): The cache for past key values.
        """

        decoder_outputs: Seq2SeqModelOutput = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=sequence_embedding,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        embeddings = decoder_outputs.last_hidden_state

        if use_cache:
            return embeddings, decoder_outputs.past_key_values
        return embeddings


class SemanticIDEncoderModule(torch.nn.Module):
    """
    This is an in-house replication of the encoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        encoder: transformers.PreTrainedModel,
    ) -> None:
        """
        Initialize the SemanticIDEncoderModule module.

        Paremeters:
        encoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        """
        super().__init__()

        self.encoder = encoder
        embedding_table_dim = find_module_shape(self.encoder, "embed_tokens")
        num_embeddings, embedding_dim = embedding_table_dim

        self.num_embeddings_per_hierarchy = num_embeddings
        self.embedding_dim = embedding_dim
        # TODO (clark): take care of chunky position encoding

        # deleting embedding table in the encoder to save space
        delete_module(self.encoder, "embed_tokens")
        delete_module(self.encoder, "shared")
        reset_parameters(self.encoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
    ) -> torch.Tensor:

        encoder_output = self.encoder(
            inputs_embeds=sequence_embedding,
            attention_mask=attention_mask,
        )
        embeddings = encoder_output.last_hidden_state
        return embeddings


# TODO (clark): this is a T5 specific implementation
# this class is used for bloating the mlp layers in the encoder and decoder
# original T5 implementation only has one layer
class T5MultiLayerFF(nn.Module):
    def __init__(self, config: T5Config, num_layers: int):
        """
        Initialize the T5MultiLayerFF module.
        This module is a multi-layer feed-forward network (MLP) used in the T5 model.
        It consists of a series of linear layers with ReLU activation and dropout.
        And it also includes layer normalization and residual connections.
        Parameters:
            config (T5Config): The T5 configuration object.
            num_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        self.mlp = MLP(
            input_dim=config.d_model,
            output_dim=config.d_model,
            hidden_dim_list=[config.d_ff for _ in range(num_layers)],
            activation=nn.ReLU,
            dropout=config.dropout_rate,
        )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the T5MultiLayerFF module.
        Parameters:
            hidden_states (torch.Tensor): The input hidden states for the MLP.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class BMTVEmbeddingWrapper(nn.Module):
    """
    Behavior-Modulated Triple-View embedding wrapper.

    Converts a flat (B, N*(1+H)) token sequence into (B, 3N, d_model) by
    building three strictly orthogonal views per item (shared, text, image),
    gating them with a 3-way behavior gate, and injecting item-level positional
    and modality-type encodings before the sequence enters the T5 encoder.

    Assumes num_hierarchies == 3 with SID levels ordered:
        L0 = shared (e_f),  L1 = text (e_t),  L2 = image (e_i)
    """

    def __init__(self, d_model: int, max_seq_len: int = 256) -> None:
        super().__init__()
        self.gating_net = nn.Linear(d_model, 3, bias=True)
        self.item_pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.modality_type_embedding = nn.Embedding(3, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, N*(1+H)) raw IDs
        attention_mask: torch.Tensor,   # (B, N*(1+H))
        embedding_table: nn.Embedding,
        stride: int,                    # = 1 + num_hierarchies = 4
        behavior_offset: int,
        sid_level_offsets: List[int],   # [off_L0, off_L1, off_L2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs_embeds : (B, 3N, d_model)
            encoder_mask  : (B, 3N)
        """
        B, L = input_ids.shape
        N = L // stride

        # Reshape into per-item groups: (B, N, stride)
        ids = input_ids.reshape(B, N, stride)
        item_mask = attention_mask.reshape(B, N, stride)[:, :, 0]  # (B, N)

        # Clamp -1 padding tokens to 0 before adding offsets
        b_ids = ids[:, :, 0].clamp(min=0)   # behavior
        f_ids = ids[:, :, 1].clamp(min=0)   # SID L0 (shared)
        t_ids = ids[:, :, 2].clamp(min=0)   # SID L1 (text)
        i_ids = ids[:, :, 3].clamp(min=0)   # SID L2 (image)

        # Global indices into the unified embedding table; 0 for padded items
        b_global = (b_ids + behavior_offset)       * item_mask  # (B, N)
        f_global = (f_ids + sid_level_offsets[0])  * item_mask
        t_global = (t_ids + sid_level_offsets[1])  * item_mask
        i_global = (i_ids + sid_level_offsets[2])  * item_mask

        e_B = embedding_table(b_global)   # (B, N, d)
        e_f = embedding_table(f_global)
        e_t = embedding_table(t_global)
        e_i = embedding_table(i_global)

        # --- Step 1: Triple-view construction (orthogonal decoupling) ---
        V_shared = e_f   # (B, N, d) — macro shared semantics
        V_text   = e_t   # (B, N, d) — text-specific residual
        V_img    = e_i   # (B, N, d) — image-specific residual

        # --- Step 2: 3-way behavior gating ---
        gate = torch.softmax(self.gating_net(e_B), dim=-1)  # (B, N, 3)
        V_shared_g = gate[:, :, 0:1] * V_shared   # (B, N, d)
        V_text_g   = gate[:, :, 1:2] * V_text     # (B, N, d)
        V_img_g    = gate[:, :, 2:3] * V_img      # (B, N, d)

        # --- Step 3: 3N sequence assembly + positional encoding ---
        # [V_shared_1, V_text_1, V_img_1, V_shared_2, ...]  →  (B, 3N, d)
        stacked     = torch.stack([V_shared_g, V_text_g, V_img_g], dim=2)  # (B, N, 3, d)
        interleaved = stacked.reshape(B, 3 * N, -1)                         # (B, 3N, d)

        # Item-level PE: all three views of item k share position k → 0,0,0,1,1,1,...
        item_pos = (
            torch.arange(N, device=input_ids.device)
            .unsqueeze(0).expand(B, -1)
            .repeat_interleave(3, dim=1)
        )  # (B, 3N)
        pos_emb = self.item_pos_embedding(item_pos)   # (B, 3N, d)

        # Modality TE: 0=shared, 1=text, 2=image → repeating pattern 0,1,2,0,1,2,...
        mod_ids = (
            torch.arange(3 * N, device=input_ids.device) % 3
        ).unsqueeze(0).expand(B, -1)  # (B, 3N)
        mod_emb = self.modality_type_embedding(mod_ids)   # (B, 3N, d)

        inputs_embeds = interleaved + pos_emb + mod_emb   # (B, 3N, d)

        # Encoder mask: each item contributes 3 tokens
        encoder_mask = item_mask.repeat_interleave(3, dim=1)   # (B, 3N)

        return inputs_embeds, encoder_mask


class SemanticIDMultiBehaviorEncoderDecoder(SemanticIDEncoderDecoder):
    """
    Multi-behavior TIGER model.

    Extends SemanticIDEncoderDecoder so that each item in the sequence is
    represented by (1 + num_hierarchies) tokens:
        [behavior_token, SID_L1, SID_L2, ..., SID_LH]

    The embedding table is extended by num_behaviors slots placed at indices
    [H * num_embeddings_per_hierarchy, ..., H * num_embeddings_per_hierarchy + num_behaviors).

    Decoder produces num_hierarchies + 1 predictions per item:
        position 0 → behavior class  (via behavior_mlp)
        position h+1 → SID level h   (via decoder_mlp[h])

    Evaluation tracks (reported via MultiBehaviorSIDRetrievalEvaluator):
        Track 1: GT behavior == buy → SID HR/NDCG
        Track 2: Conditioned on GT behavior → SID HR/NDCG
        Track 3: Free generation → behavior accuracy + joint HR/NDCG
    """

    def __init__(
        self,
        num_behaviors: int = 4,
        buy_behavior_id: int = 3,
        codebook_sizes: Optional[List[int]] = None,
        codebook_init_path: Optional[str] = None,
        use_bmtv: bool = False,
        bmtv_max_seq_len: int = 256,
        use_item_pe: bool = False,
        item_pe_max_seq_len: int = 256,
        focal_behavior: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[List[float]] = None,
        behavior_weighted_sid: bool = False,
        sid_behavior_weights: Optional[List[float]] = None,
        use_semantic_align: bool = False,
        semantic_align_weight: float = 1.0,
        align_loss_type: str = "mse",
        align_temperature: float = 0.07,
        use_semantic_rerank: bool = False,
        rerank_weight: float = 0.5,
        audit_tf_dump: bool = False,
        audit_tf_path: str = "/scratch/yw8866/rec-tmall/chain_tf_audit.pt",
        audit_tf_max_events: int = 200000,
        audit_tf_pv_cap: int = 100000,
        audit_tf_pv_id: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_behaviors = num_behaviors

        # Chain teacher-forced per-level audit (0b reference; default OFF, chain-only). When on,
        # the test loop records, per event, the rank of t* in P(t|GT f) and i* in P(i|GT f,GT t)
        # from the teacher-forced decoder, so 0b can compare MFD's shared-h_f reads against the
        # chain's dedicated-h_f/h_t reads. Not used by MFD (it has its own richer mfd_audit_dump).
        self.audit_tf_dump = audit_tf_dump
        self.audit_tf_path = audit_tf_path
        self.audit_tf_max_events = audit_tf_max_events
        self.audit_tf_pv_cap = audit_tf_pv_cap
        self.audit_tf_pv_id = audit_tf_pv_id
        self._tf_cols = collections.defaultdict(list) if audit_tf_dump else None
        self._tf_n = 0
        self._tf_pv = 0
        self.buy_behavior_id = buy_behavior_id

        # ------------------------------------------------------------------
        # Per-level codebook sizes and cumulative SID offsets
        # ------------------------------------------------------------------
        if codebook_sizes is not None:
            if len(codebook_sizes) != self.num_hierarchies:
                raise ValueError(
                    f"codebook_sizes must have {self.num_hierarchies} entries, "
                    f"got {len(codebook_sizes)}"
                )
            self.sid_level_sizes: List[int] = list(codebook_sizes)
        else:
            self.sid_level_sizes = [self.num_embeddings_per_hierarchy] * self.num_hierarchies

        # cumulative base offsets per SID level: [0, K0, K0+K1, ...]
        self.sid_level_offsets: List[int] = [
            sum(self.sid_level_sizes[:h]) for h in range(self.num_hierarchies)
        ]
        # behavior tokens sit immediately after all SID entries
        self.behavior_offset: int = sum(self.sid_level_sizes)
        total_vocab = self.behavior_offset + num_behaviors

        # ------------------------------------------------------------------
        # Build / reinitialise the embedding table
        # ------------------------------------------------------------------
        if codebook_init_path is not None:
            # Inject codebook geometry via orthogonal up-projection
            e_init, detected_sizes = build_codebook_init_embedding(
                checkpoint_path=codebook_init_path,
                d_model=self.embedding_dim,
                num_behaviors=num_behaviors,
            )
            if codebook_sizes is not None and list(codebook_sizes) != detected_sizes:
                raise ValueError(
                    f"codebook_sizes {codebook_sizes} does not match sizes "
                    f"found in checkpoint {detected_sizes}"
                )
            # Sync sid_level_sizes in case codebook_sizes was None
            self.sid_level_sizes = detected_sizes
            self.sid_level_offsets = [
                sum(self.sid_level_sizes[:h]) for h in range(self.num_hierarchies)
            ]
            self.behavior_offset = sum(self.sid_level_sizes)
            total_vocab = self.behavior_offset + num_behaviors

            new_table = torch.nn.Embedding(total_vocab, self.embedding_dim)
            with torch.no_grad():
                new_table.weight.copy_(e_init)
        else:
            # Random init: extend the existing table to fit the new vocab layout
            new_table = torch.nn.Embedding(total_vocab, self.embedding_dim)
            with torch.no_grad():
                # Copy old SID entries level by level (old layout: h*old_cbs blocks)
                old_cbs = self.num_embeddings_per_hierarchy
                old_weight = self.item_sid_embedding_table_encoder.weight.data
                for h, (old_off, new_off, k) in enumerate(
                    zip(
                        [h * old_cbs for h in range(self.num_hierarchies)],
                        self.sid_level_offsets,
                        self.sid_level_sizes,
                    )
                ):
                    copy_k = min(k, old_cbs, old_weight.shape[0] - old_off)
                    if copy_k > 0:
                        new_table.weight[new_off : new_off + copy_k] = (
                            old_weight[old_off : old_off + copy_k]
                        )

        self.item_sid_embedding_table_encoder = new_table

        # ------------------------------------------------------------------
        # Replace decoder MLP heads with correct per-level output sizes
        # ------------------------------------------------------------------
        self.decoder.decoder_mlp = torch.nn.ModuleList([
            torch.nn.Linear(self.embedding_dim, k, bias=False)
            for k in self.sid_level_sizes
        ])

        # Linear head that predicts the behavior type
        self.behavior_mlp = torch.nn.Linear(self.embedding_dim, num_behaviors, bias=False)

        # ------------------------------------------------------------------
        # BM-TV embedding wrapper (optional, controlled by use_bmtv flag)
        # ------------------------------------------------------------------
        self.use_bmtv = use_bmtv
        if use_bmtv:
            if self.num_hierarchies != 3:
                raise ValueError(
                    f"BMTVEmbeddingWrapper requires num_hierarchies == 3 "
                    f"(shared / text / image SID levels), got {self.num_hierarchies}"
                )
            self.bmtv_wrapper = BMTVEmbeddingWrapper(
                d_model=self.embedding_dim,
                max_seq_len=bmtv_max_seq_len,
            )
            # sep_token is not called in the BM-TV encoder path.
            # Nullify it so DDP does not flag it as an unused parameter and crash.
            self.register_parameter('sep_token', None)

        # ------------------------------------------------------------------
        # Explicit item-level PE + modality TE for the flat sequence
        # (optional, controlled by use_item_pe flag, mutually exclusive with use_bmtv)
        # ------------------------------------------------------------------
        if use_item_pe and use_bmtv:
            raise ValueError("use_item_pe and use_bmtv are mutually exclusive")
        self.use_item_pe = use_item_pe
        if use_item_pe:
            stride = 1 + self.num_hierarchies
            # item_pos_embedding: items share one slot → index 0..N-1
            self.item_pos_embedding = nn.Embedding(item_pe_max_seq_len, self.embedding_dim)
            # modality_type_embedding: one index per within-item slot (B=0, L0=1, L1=2, ..., LH=stride-1)
            self.modality_type_embedding = nn.Embedding(stride, self.embedding_dim)
            # sep_token bypassed; nullify it to avoid DDP unused-parameter crash.
            self.register_parameter('sep_token', None)

        # ------------------------------------------------------------------
        # Loss rebalancing for the behavior class imbalance (optional)
        # ------------------------------------------------------------------
        # (1) Focal loss on the behavior head — focuses gradient on hard/rare
        #     behaviors (buy/cart/fav) instead of the dominant pv.
        self.focal_behavior = focal_behavior
        if focal_behavior:
            self.behavior_loss_function = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.behavior_loss_function = None

        # (2) Behavior-weighted SID loss — weight each item's SID loss by its TARGET
        #     behavior so rare-behavior interactions (buy) get more SID gradient.
        self.behavior_weighted_sid = behavior_weighted_sid
        if behavior_weighted_sid:
            if sid_behavior_weights is None:
                sid_behavior_weights = [1.0] * num_behaviors
            if len(sid_behavior_weights) != num_behaviors:
                raise ValueError(
                    f"sid_behavior_weights must have {num_behaviors} entries "
                    f"(one per behavior), got {len(sid_behavior_weights)}"
                )
            self.register_buffer(
                "sid_behavior_weights",
                torch.tensor(sid_behavior_weights, dtype=torch.float),
            )

        # ------------------------------------------------------------------
        # Two-View Semantic Alignment + Retrieval (optional)
        # ------------------------------------------------------------------
        # Predict the next item's MD-RQ-VAE reconstruction views v_t=e_f+e_t (text)
        # and v_i=e_f+e_i (image) from the decoder's item-position hidden state.
        #   - use_semantic_align: add an MSE align loss to a stop-grad target (train).
        #   - use_semantic_rerank: rerank the free-gen candidate pool by cosine of the
        #     predicted (q_t, q_i) to each candidate's (v_t, v_i) (inference).
        # Requires num_hierarchies == 3 (shared/text/image levels).
        self.use_semantic_align = use_semantic_align
        self.semantic_align_weight = semantic_align_weight
        if align_loss_type not in ("mse", "contrastive"):
            raise ValueError(f"align_loss_type must be 'mse' or 'contrastive', got {align_loss_type}")
        self.align_loss_type = align_loss_type
        self.align_temperature = align_temperature
        self.use_semantic_rerank = use_semantic_rerank
        self.rerank_weight = rerank_weight
        if use_semantic_align or use_semantic_rerank:
            if self.num_hierarchies != 3:
                raise ValueError(
                    "Two-View semantic align/rerank requires num_hierarchies == 3 "
                    f"(shared/text/image), got {self.num_hierarchies}"
                )
            self.align_head_text = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
            self.align_head_image = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

    # ------------------------------------------------------------------
    # Offset helpers
    # ------------------------------------------------------------------

    def _mb_offset_pattern(self, num_cols: int, device: torch.device) -> torch.Tensor:
        """(1+H)-periodic offset: [behavior_offset, off_L0, off_L1, ..., off_LH-1, ...]"""
        period = [self.behavior_offset] + self.sid_level_offsets
        stride = 1 + self.num_hierarchies
        num_repeats = (num_cols + stride - 1) // stride
        return torch.tensor(period * num_repeats, dtype=torch.long, device=device)[:num_cols]

    def _sid_two_view(self, sid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map SID indices to the MD-RQ-VAE reconstruction views (requires H==3):
            v_t = e_f + e_t (text latent),  v_i = e_f + e_i (image latent)
        sid: (..., 3) -> (v_t, v_i), each (..., embedding_dim).
        """
        table = self.item_sid_embedding_table_encoder
        o0, o1, o2 = self.sid_level_offsets
        sid = sid.long().clamp(min=0)
        e_f = table(sid[..., 0] + o0)
        e_t = table(sid[..., 1] + o1)
        e_i = table(sid[..., 2] + o2)
        return e_f + e_t, e_f + e_i

    def _blend_rerank(self, marginal: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """
        Per-sample min-max normalize the decoder marginal score and the semantic
        similarity, then blend: score = (1-w)*marginal + w*sim  (w = rerank_weight).
        w=0 reproduces the decoder ranking (baseline); w=1 is pure semantic retrieval.
        """
        def _norm(x: torch.Tensor) -> torch.Tensor:
            mn = x.min(dim=1, keepdim=True).values
            mx = x.max(dim=1, keepdim=True).values
            return (x - mn) / (mx - mn).clamp(min=1e-8)

        w = self.rerank_weight
        return (1.0 - w) * _norm(marginal) + w * _norm(sim)

    def _contrastive_align(
        self, pred: torch.Tensor, tgt: torch.Tensor, gt_sid: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE align loss: the predicted query `pred` should rank its own GT item's
        view `tgt` above the other items in the batch (in-batch negatives), using cosine
        similarity / temperature — the same geometry used by the rerank at inference.

        In-batch items that share the anchor's exact SID tuple are masked out (they are
        the same item — false negatives; collision rate ~36%).
        """
        B = pred.size(0)
        pred_n = torch.nn.functional.normalize(pred, dim=-1)
        tgt_n = torch.nn.functional.normalize(tgt.detach(), dim=-1)
        logits = (pred_n @ tgt_n.t()) / self.align_temperature  # (B, B)
        same = (gt_sid.unsqueeze(1) == gt_sid.unsqueeze(0)).all(dim=-1)  # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=pred.device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))
        labels = torch.arange(B, device=pred.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Forward pass overrides
    # ------------------------------------------------------------------

    def encoder_forward_pass(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_bmtv:
            # BM-TV path: produces (B, 3N, d_model) + matching (B, 3N) mask.
            # sep_token is intentionally skipped; PE/TE in the wrapper carry
            # the structural signals.
            inputs_embeds, attention_mask = self.bmtv_wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embedding_table=self.item_sid_embedding_table_encoder,
                stride=1 + self.num_hierarchies,
                behavior_offset=self.behavior_offset,
                sid_level_offsets=self.sid_level_offsets,
            )
        else:
            # Standard MB path: (1+H)-periodic offset lookup → (B, N*(1+H), d)
            offsets = self._mb_offset_pattern(input_ids.shape[1], input_ids.device)
            shifted = (input_ids + offsets) * attention_mask
            inputs_embeds = self.get_embedding_table("encoder")(shifted)

            if self.use_item_pe:
                # Inject item-level PE and modality TE into the flat sequence.
                # Sequence layout: [B_1, L0_1, L1_1, ..., B_2, L0_2, ...]
                #   item_pos: 0,0,...(stride times),1,1,...  →  ties all tokens of item k
                #   mod_ids:  0,1,2,...,stride-1 repeating  →  encodes within-item slot type
                B_sz, L = input_ids.shape
                stride = 1 + self.num_hierarchies
                N = L // stride
                item_pos = (
                    torch.arange(N, device=input_ids.device)
                    .repeat_interleave(stride)
                    .unsqueeze(0).expand(B_sz, -1)
                )  # (B, L)  — 0,0,0,0,1,1,1,1,...
                mod_ids = (
                    torch.arange(L, device=input_ids.device) % stride
                ).unsqueeze(0).expand(B_sz, -1)  # (B, L)  — 0,1,2,3,0,1,2,3,...
                inputs_embeds = (
                    inputs_embeds
                    + self.item_pos_embedding(item_pos)
                    + self.modality_type_embedding(mod_ids)
                )

            if self.sep_token is not None:
                inputs_embeds, attention_mask = self._inject_sep_token_between_sids(
                    id_embeddings=inputs_embeds,
                    attention_mask=attention_mask,
                    sep_token=self.sep_token,
                    num_hierarchies=(1 + self.num_hierarchies),
                )

        if user_id is not None and self.user_embedding is not None:
            user_id = user_id[:, 0]
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )
            inputs_embeds = torch.cat([user_embeds.unsqueeze(1), inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [torch.ones(attention_mask.size(0), 1, device=attention_mask.device),
                 attention_mask],
                dim=1,
            )

        encoder_output = self.encoder(
            sequence_embedding=inputs_embeds,
            attention_mask=attention_mask,
        )
        return encoder_output, attention_mask

    def decoder_forward_pass(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        future_ids: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask_for_encoder: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values=None,
    ) -> torch.Tensor:
        if future_ids is not None:
            offsets = self._mb_offset_pattern(future_ids.shape[1], future_ids.device)
            ones_mask = torch.ones_like(future_ids)
            shifted_future = future_ids + offsets
            shifted_future = shifted_future * ones_mask
            inputs_embeds = self.get_embedding_table("decoder")(shifted_future)

            if not self._is_kv_cache_valid(kv_cache=past_key_values):
                inputs_embeds = torch.cat(
                    [self.decoder.bos_token.unsqueeze(0).expand(future_ids.size(0), 1, -1),
                     inputs_embeds],
                    dim=1,
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [torch.ones(future_ids.size(0), 1, device=future_ids.device),
                         attention_mask],
                        dim=1,
                    )
            else:
                inputs_embeds = inputs_embeds[:, -1:, :]
        else:
            inputs_embeds = self.decoder.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

        return self.decoder(
            sequence_embedding=inputs_embeds,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask_for_encoder,
            encoder_output=encoder_output,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_multibehavior(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor = None,
    ):
        """
        Beam-search generation that produces (1 + num_hierarchies) tokens per item:
            [:, :, 0]   = predicted behavior
            [:, :, 1:]  = predicted SID levels

        Returns:
            generated_ids:   (B, top_k, 1+H)
            marginal_probs:  (B, top_k)
        """
        encoder_output, encoder_attention_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )

        batch_size = input_ids.size(0)
        past_key_values = EncoderDecoderCache(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )

        # ---- step 0: predict behavior (no previous generated tokens) ----
        decoder_output, past_key_values = self.decoder_forward_pass(
            future_ids=None,
            encoder_output=encoder_output,
            attention_mask_for_encoder=encoder_attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        behavior_logits = self.behavior_mlp(decoder_output[:, -1, :])  # (B, num_behaviors)

        # Two-View rerank query: predict the next item's reconstruction views from the
        # item-summary hidden state (after BOS, before committing to behavior/SIDs).
        if self.use_semantic_rerank:
            h_item = decoder_output[:, -1, :]  # (B, d)
            rerank_q_t = self.align_head_text(h_item)   # (B, d)
            rerank_q_i = self.align_head_image(h_item)  # (B, d)

        # top-k over behavior (small vocab — simple selection, no prefix check)
        top_k_b = min(self.top_k_for_generation, self.num_behaviors)
        behavior_probs = torch.nn.functional.softmax(behavior_logits, dim=-1)
        proba_topk, idx_topk = torch.topk(behavior_probs, top_k_b, dim=-1)  # (B, top_k_b)

        generated_ids = idx_topk.unsqueeze(-1)      # (B, top_k_b, 1)
        marginal_log_prob = proba_topk               # (B, top_k_b)

        # Pad to top_k_for_generation so the SID loop always sees exactly top_k beams.
        # Padded beams carry probability 0 and can never win any topk selection.
        if top_k_b < self.top_k_for_generation:
            pad = self.top_k_for_generation - top_k_b
            generated_ids = torch.cat(
                [generated_ids, generated_ids.new_zeros(batch_size, pad, 1)], dim=1
            )
            marginal_log_prob = torch.cat(
                [marginal_log_prob, torch.zeros(batch_size, pad, device=marginal_log_prob.device)],
                dim=1,
            )

        # reset cache after expanding beams from B → B*top_k
        past_key_values = EncoderDecoderCache(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )

        # ---- steps 1..H: predict SID levels ----
        for hierarchy in range(self.num_hierarchies):
            step = hierarchy + 1  # position in the (1+H)-token block
            squeezed = generated_ids.reshape(-1, step)  # (B*top_k, step)

            repeated_encoder_output = encoder_output.repeat_interleave(
                self.top_k_for_generation, dim=0
            )
            repeated_encoder_mask = encoder_attention_mask.repeat_interleave(
                self.top_k_for_generation, dim=0
            )

            decoder_output, past_key_values = self.decoder_forward_pass(
                future_ids=squeezed,
                encoder_output=repeated_encoder_output,
                attention_mask_for_encoder=repeated_encoder_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            candidate_logits = self.decoder.decoder_mlp[hierarchy](
                decoder_output[:, -1, :]
            )  # (B*top_k, sid_level_sizes[hierarchy])

            generated_ids, marginal_log_prob, past_key_values = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                past_key_values=past_key_values,
                hierarchy=step,
                batch_size=batch_size,
                num_emb_override=self.sid_level_sizes[hierarchy],
            )

        # Two-View rerank: reorder the candidate pool by cosine of the predicted views
        # (rerank_q_t, rerank_q_i) to each candidate's (v_t, v_i). The evaluator then
        # takes top-K by this blended score, so widen top_k_for_generation into a pool.
        if self.use_semantic_rerank:
            cand_vt, cand_vi = self._sid_two_view(generated_ids[:, :, 1:])  # (B, pool, d)
            cos = torch.nn.functional.cosine_similarity
            sim = 0.5 * (
                cos(rerank_q_t.unsqueeze(1), cand_vt, dim=-1)
                + cos(rerank_q_i.unsqueeze(1), cand_vi, dim=-1)
            )  # (B, pool)
            marginal_log_prob = self._blend_rerank(marginal_log_prob, sim)

        return generated_ids, marginal_log_prob  # (B, top_k, 1+H), (B, top_k)

    def generate_conditioned(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        gt_behavior: torch.Tensor,
        user_id: torch.Tensor = None,
    ):
        """
        Track-2 generation: force the behavior token to gt_behavior, then beam-search SIDs.

        Args:
            gt_behavior: (B,) long tensor of GT behavior indices.

        Returns:
            generated_ids:  (B, top_k, H)  — SIDs only (no behavior prefix)
            marginal_probs: (B, top_k)
        """
        encoder_output, encoder_attention_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )

        batch_size = input_ids.size(0)
        past_key_values = EncoderDecoderCache(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )

        # seed generated_ids with the GT behavior — shape (B, 1, 1)
        generated_ids = gt_behavior.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, self.top_k_for_generation, 1
        )  # (B, top_k, 1)
        marginal_log_prob = torch.ones(
            batch_size, self.top_k_for_generation, device=gt_behavior.device
        )  # uniform — behavior is given

        # SID beam search (steps 1..H), same as generate_multibehavior after step 0
        for hierarchy in range(self.num_hierarchies):
            step = hierarchy + 1
            squeezed = generated_ids.reshape(-1, step)  # (B*top_k, step)

            repeated_encoder_output = encoder_output.repeat_interleave(
                self.top_k_for_generation, dim=0
            )
            repeated_encoder_mask = encoder_attention_mask.repeat_interleave(
                self.top_k_for_generation, dim=0
            )

            decoder_output, past_key_values = self.decoder_forward_pass(
                future_ids=squeezed,
                encoder_output=repeated_encoder_output,
                attention_mask_for_encoder=repeated_encoder_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            candidate_logits = self.decoder.decoder_mlp[hierarchy](
                decoder_output[:, -1, :]
            )

            generated_ids, marginal_log_prob, past_key_values = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                past_key_values=past_key_values,
                hierarchy=step,
                batch_size=batch_size,
                num_emb_override=self.sid_level_sizes[hierarchy],
            )

        # strip behavior prefix, return SIDs only
        return generated_ids[:, :, 1:], marginal_log_prob  # (B, top_k, H), (B, top_k)

    # ------------------------------------------------------------------
    # Training / eval steps
    # ------------------------------------------------------------------

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if label_data is None:
            generated_ids, marginal_probs = self.generate_multibehavior(
                attention_mask=model_input.mask,
                **{
                    self.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                },
            )
            return generated_ids, 0

        fut_ids = None
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            fut_ids = curr_label.reshape(model_input.mask.size(0), -1)
        # fut_ids shape: (B, 1 + num_hierarchies)

        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )
        # Decoder receives [BOS, behavior, SID_L1, ..., SID_LH]; output shape (B, 2+H, embed_dim).
        # Drop the last position (prediction for position after the last label).
        model_output = model_output[:, :-1]  # (B, 1+H, embed_dim)

        # behavior loss at position 0 (focal if enabled, else plain CE)
        behavior_logits = self.behavior_mlp(model_output[:, 0])  # (B, num_behaviors)
        behavior_target = fut_ids[:, 0].long()
        behavior_loss_fn = self.behavior_loss_function or self.loss_function
        loss = behavior_loss_fn(input=behavior_logits, target=behavior_target)

        # Per-example SID weights keyed by the TARGET behavior (behavior-weighted SID).
        if self.behavior_weighted_sid:
            sid_weights = self.sid_behavior_weights.to(behavior_target.device)[behavior_target]  # (B,)
            weight_denom = sid_weights.sum().clamp(min=1e-8)

        # SID losses at positions 1..H
        for hierarchy in range(self.num_hierarchies):
            sid_logits = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy + 1])
            sid_target = fut_ids[:, hierarchy + 1].long()
            if self.behavior_weighted_sid:
                # weighted mean over the batch (normalized by sum of weights)
                per_example_ce = torch.nn.functional.cross_entropy(
                    sid_logits, sid_target, reduction="none"
                )  # (B,)
                loss += (sid_weights * per_example_ce).sum() / weight_denom
            else:
                loss += self.loss_function(input=sid_logits, target=sid_target)

        # Two-View semantic alignment loss: predict the next item's reconstruction
        # views (v_t=e_f+e_t, v_i=e_f+e_i) from the item-summary hidden state (pos 0,
        # after BOS) and regress to a stop-grad target.
        if self.use_semantic_align:
            h_item = model_output[:, 0]  # (B, d)
            pred_vt = self.align_head_text(h_item)
            pred_vi = self.align_head_image(h_item)
            with torch.no_grad():
                tgt_vt, tgt_vi = self._sid_two_view(fut_ids[:, 1:])
            if self.align_loss_type == "contrastive":
                gt_sid = fut_ids[:, 1:].long()
                align_loss = (
                    self._contrastive_align(pred_vt, tgt_vt, gt_sid)
                    + self._contrastive_align(pred_vi, tgt_vi, gt_sid)
                )
            else:  # mse
                align_loss = (
                    torch.nn.functional.mse_loss(pred_vt, tgt_vt)
                    + torch.nn.functional.mse_loss(pred_vi, tgt_vi)
                )
            loss = loss + self.semantic_align_weight * align_loss

        return model_output, loss

    @torch.no_grad()
    def _update_sid_probe(
        self,
        generated_ids: torch.Tensor,   # (B, top_k, 1+H)
        marginal_probs: torch.Tensor,  # (B, top_k)
        labels: torch.Tensor,          # (B, 1+H)
    ) -> None:
        """
        Semantic near-miss probe. Among MISS cases (GT SID tuple not in the top-k
        generated SIDs), measure how close the best-beam predicted SID is to the GT
        SID in the MD-RQ-VAE reconstruction space — cosine similarity of v_t=e_f+e_t
        and v_i=e_f+e_i — versus a random in-batch item. Accumulated per behavior +
        overall into the evaluator's `probe_*` metrics.

          near-miss  : pred_sim >> rand_sim  -> semantic retrieval could recover GT
          unpredictable: pred_sim ≈ rand_sim -> the miss is essentially random

        NOTE: rebalancing (user_drop / event_downsample) is training-only; eval/test
        stay at the natural ~93% pv distribution for every run. So `overall` is the same
        mix across runs (and is pv-dominated, hence uninformative about rare behaviors)
        — read the per-behavior groups (buy/cart/fav) for the rare-behavior signal.
        """
        if self.num_hierarchies != 3:
            return  # property e_f+e_t / e_f+e_i is defined for shared/text/image only

        device = labels.device
        B = labels.size(0)
        table = self.item_sid_embedding_table_encoder
        o0, o1, o2 = self.sid_level_offsets

        gt_behavior = labels[:, 0].long()
        gt_sid = labels[:, 1:].long().clamp(min=0)               # (B, 3)
        pred_all = generated_ids[:, :, 1:].long().clamp(min=0)   # (B, k, 3)
        best = marginal_probs.argmax(dim=1)                      # (B,)
        pred_sid = pred_all[torch.arange(B, device=device), best]  # (B, 3)

        # miss = GT SID tuple absent from every top-k beam
        miss = ~torch.all(pred_all == gt_sid.unsqueeze(1), dim=2).any(dim=1)  # (B,)

        def sem(sid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            e_f = table(sid[..., 0] + o0)
            e_t = table(sid[..., 1] + o1)
            e_i = table(sid[..., 2] + o2)
            return e_f + e_t, e_f + e_i  # (v_t, v_i)

        vt_gt, vi_gt = sem(gt_sid)
        vt_pr, vi_pr = sem(pred_sid)
        perm = torch.randperm(B, device=device)
        cos = torch.nn.functional.cosine_similarity
        sim_pred = 0.5 * (cos(vt_pr, vt_gt, dim=-1) + cos(vi_pr, vi_gt, dim=-1))      # (B,)
        sim_rand = 0.5 * (cos(vt_gt[perm], vt_gt, dim=-1) + cos(vi_gt[perm], vi_gt, dim=-1))

        groups = [("overall", torch.ones(B, dtype=torch.bool, device=device))]
        for bid, bname in self.evaluator.behavior_names.items():
            groups.append((bname, gt_behavior == bid))

        for gname, gmask in groups:
            total = int(gmask.sum().item())
            if total > 0:
                self.evaluator.metrics[f"probe_{gname}_miss_frac"].update(
                    float((gmask & miss).sum().item()), total
                )
            m = gmask & miss
            cnt = int(m.sum().item())
            if cnt > 0:
                self.evaluator.metrics[f"probe_{gname}_miss_pred_sim"].update(
                    float(sim_pred[m].sum().item()), cnt
                )
                self.evaluator.metrics[f"probe_{gname}_miss_rand_sim"].update(
                    float(sim_rand[m].sum().item()), cnt
                )

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate,
    ):
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]

        if self.audit_tf_dump:
            self._tf_audit_step(model_input, label_data)
            return

        _, loss = self.model_step(model_input=model_input, label_data=label_data)

        # free generation for Track 1 and Track 3
        generated_ids, marginal_probs = self.generate_multibehavior(
            attention_mask=model_input.mask,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        labels = list(label_data.labels.values())[0].to(marginal_probs.device)
        # labels shape: (B * (1+H),) flattened; reshape to (B, 1+H)
        labels = labels.reshape(model_input.mask.size(0), -1)

        # Track 2: conditioned generation
        gt_behavior = labels[:, 0]
        generated_ids_t2, marginal_probs_t2 = self.generate_conditioned(
            attention_mask=model_input.mask,
            gt_behavior=gt_behavior,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            labels=labels,
            marginal_probs_t2=marginal_probs_t2,
            generated_ids_t2=generated_ids_t2,
        )

        # Semantic near-miss probe (no-op unless probe_sid_distance=True on the evaluator).
        if getattr(self.evaluator, "probe_sid_distance", False):
            self._update_sid_probe(generated_ids, marginal_probs, labels)

        loss_to_aggregate(loss)

    @torch.no_grad()
    def _tf_audit_step(self, model_input, label_data):
        """Chain teacher-forced per-level audit (0b reference). Records rank of t* in P(t|GT f)
        and i* in P(i|GT f,GT t) from the causal decoder, plus GT behavior, per event."""
        if self._tf_n >= self.audit_tf_max_events:
            return
        fut_ids = list(label_data.labels.values())[0].reshape(model_input.mask.size(0), -1)
        fut_ids = fut_ids.to(model_input.mask.device)
        # model_step forward returns model_output[:, :-1] = (B, 1+H, d):
        #   [:,2] = h after GT f -> predicts t ;  [:,3] = h after GT t -> predicts i
        mo, _ = self.model_step(model_input=model_input, label_data=label_data)
        ar = torch.arange(mo.size(0), device=mo.device)
        gt_t = fut_ids[:, 2].long(); gt_i = fut_ids[:, 3].long()
        t_logits = self.decoder.decoder_mlp[1](mo[:, 2])
        i_logits = self.decoder.decoder_mlp[2](mo[:, 3])
        tf_t_rank = (t_logits > t_logits[ar, gt_t].unsqueeze(1)).sum(1)
        tf_i_rank = (i_logits > i_logits[ar, gt_i].unsqueeze(1)).sum(1)
        gt_b = fut_ids[:, 0].long()
        keep = torch.zeros(gt_b.shape[0], dtype=torch.bool)
        for idx in range(gt_b.shape[0]):
            if self._tf_n >= self.audit_tf_max_events:
                break
            if gt_b[idx].item() == self.audit_tf_pv_id:
                if self._tf_pv >= self.audit_tf_pv_cap:
                    continue
                self._tf_pv += 1
            keep[idx] = True
            self._tf_n += 1
        if keep.any():
            self._tf_cols["gt_b"].append(gt_b[keep].to(torch.int16).cpu())
            self._tf_cols["tf_t_rank"].append(tf_t_rank[keep].to(torch.int32).cpu())
            self._tf_cols["tf_i_rank"].append(tf_i_rank[keep].to(torch.int32).cpu())

    def on_test_epoch_end(self):
        # Non-audit runs MUST fall through to the base hook that logs test/loss + metrics.
        if not self.audit_tf_dump:
            return super().on_test_epoch_end()
        if self._tf_cols:
            cols = {k: torch.cat(v, dim=0) for k, v in self._tf_cols.items()}
            torch.save(
                {"cols": cols, "meta": {"nb": self.num_behaviors, "pv_id": self.audit_tf_pv_id,
                                        "buy_id": self.buy_behavior_id, "kind": "chain_tf"}},
                self.audit_tf_path,
            )
            print(f"[chain tf audit] wrote {cols['gt_b'].shape[0]:,} events -> {self.audit_tf_path}",
                  flush=True)
        return  # audit run: evaluator was never updated, skip metric logging
