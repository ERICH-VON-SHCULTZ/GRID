"""PBA (Position & Behavior Aware) multi-behavior generative recommender.

A faithful port of MMMB-Genrec's PBATransformer architecture into GRID's harness. The two
core ideas from `/scratch/yw8866/MMMB-Genrec/model/PBA_transformer.py` are reproduced:

  1. Position-routed experts (deterministic MoE). Each within-item token position gets its
     OWN feed-forward expert: expert 0 = special tokens (user / BOS / pad), expert 1 =
     behavior token, experts 2..(1+H) = the H SID levels. Routing is by token position, NOT
     a learned gate (so there is no router aux/z-loss).
  2. Behavior injection. On selected layers the item's behavior embedding is concatenated
     onto the FFN input, conditioning every SID token on its behavior.

Rather than re-deriving MBGen's Switch-Transformer stack, this subclasses GRID's
`SemanticIDMultiBehaviorEncoderDecoder` and swaps each T5 `T5LayerFF` for `PBAFeedForward`.
Everything else -- SID embedding tables, the (1+H)-token sequence layout, beam-search
generation, `model_step`, and the `MultiBehaviorSIDRetrievalEvaluator` -- is inherited
unchanged, so the t1/t2/t3 metrics are directly comparable to the MD-RQ-VAE tiger baseline.

The parent model builds the flat sequence and calls `self.encoder(...)` / `self.decoder(...)`;
the T5 block FFN cannot see token positions. So before each backbone call we compute the
per-token position/behavior indices for exactly that sequence and stash them on every
PBAFeedForward (all layers in one pass see the same [B, L]). This matches MBGen's
PBAEncoderRouter / PBADecoderRouter, whose position pattern is likewise deterministic.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDMultiBehaviorEncoderDecoder,
)


class PBAExpert(nn.Module):
    """A single T5-style dense FFN expert (wi -> act -> dropout -> wo), no bias.

    When behavior injection is on for the owning layer, `wi` takes d_model + behavior_dim
    inputs (the behavior embedding is concatenated upstream in PBAFeedForward).
    """

    def __init__(self, d_model, d_ff, dropout, act_fn, in_extra=0):
        super().__init__()
        self.wi = nn.Linear(d_model + in_extra, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = act_fn

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class PBAFeedForward(nn.Module):
    """Drop-in replacement for T5LayerFF with position-routed experts + behavior injection.

    forward(hidden_states) keeps T5LayerFF's signature (layernorm -> FFN -> residual), so it
    plugs into T5Block unchanged. The per-token position/behavior indices are read from
    attributes (`pos_index`, `beh_index`) set by the parent model right before the backbone
    call -- both shaped [B, L] to match hidden_states [B, L, d].
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        eps: float,
        act_fn,
        num_experts: int,
        is_sparse: bool,
        behavior_injection: bool,
        num_behaviors: int,
        behavior_embedding_dim: int,
    ):
        super().__init__()
        from transformers.models.t5.modeling_t5 import T5LayerNorm

        self.is_sparse = is_sparse
        self.behavior_injection = behavior_injection
        in_extra = behavior_embedding_dim if behavior_injection else 0

        if is_sparse:
            self.experts = nn.ModuleList(
                [
                    PBAExpert(d_model, d_ff, dropout, act_fn, in_extra)
                    for _ in range(num_experts)
                ]
            )
        else:
            self.mlp = PBAExpert(d_model, d_ff, dropout, act_fn, in_extra)

        if behavior_injection:
            # index 0 = "no behavior" (user/BOS/pad/behavior-token slots); 1..num_behaviors
            # = the item's behavior. Mirrors MBGen's nn.Embedding(num_behavior + 1, dim).
            self.behavior_embedding = nn.Embedding(
                num_behaviors + 1, behavior_embedding_dim
            )

        self.layer_norm = T5LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

        # Set per-forward by the parent model (PBAMultiBehaviorEncoderDecoder).
        self.pos_index: Optional[torch.Tensor] = None
        self.beh_index: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)

        if self.behavior_injection:
            beh_emb = self.behavior_embedding(self.beh_index)  # [B, L, beh_dim]
            normed = torch.cat([normed, beh_emb], dim=-1)  # [B, L, d + beh_dim]

        if self.is_sparse:
            out = torch.zeros_like(hidden_states)
            for e, expert in enumerate(self.experts):
                mask = self.pos_index == e  # [B, L]
                if mask.any():
                    out[mask] = expert(normed[mask]).to(out.dtype)
        else:
            out = self.mlp(normed)

        return hidden_states + self.dropout(out)


class PBAMultiBehaviorEncoderDecoder(SemanticIDMultiBehaviorEncoderDecoder):
    """MBGen PBA architecture on GRID's multi-behavior TIGER scaffolding."""

    def __init__(
        self,
        behavior_embedding_dim: int = 64,
        sparse_layers_encoder: Optional[List[int]] = None,
        sparse_layers_decoder: Optional[List[int]] = None,
        behavior_injection_encoder: Optional[List[int]] = None,
        behavior_injection_decoder: Optional[List[int]] = None,
        moe_behavior_only: bool = False,
        **kwargs,
    ) -> None:
        # MBGen has no SEP token and no separate item-PE/BM-TV path -- the position routing
        # IS the structural signal. Force those off so the sequence layout is exactly
        # [ (user), behavior, sid0..sid_{H-1}, behavior, ... ] with stride (1+H).
        kwargs["should_add_sep_token"] = False
        if kwargs.get("use_bmtv") or kwargs.get("use_item_pe"):
            raise ValueError("PBA model is incompatible with use_bmtv / use_item_pe")
        super().__init__(**kwargs)

        self.num_positions = 1 + self.num_hierarchies  # behavior + H SID levels
        self.moe_behavior_only = moe_behavior_only
        # Highest expert id the position pattern can emit (+1 for the special expert 0).
        # moe_behavior_only collapses the H SID levels onto a single expert (id 2), so only
        # experts 0..2 exist -- allocating more would leave dead params (DDP crash on plain
        # ddp). Otherwise one expert per position: 0..num_positions.
        max_expert = 2 if moe_behavior_only else self.num_positions
        self.num_experts = max_expert + 1
        self.behavior_embedding_dim = behavior_embedding_dim

        # Locate the two T5Stack modules (the things that own `.block`). The encoder is a
        # T5EncoderModel (stack at .encoder), the decoder is a bare T5Stack -- both are
        # wrapped one level deep by SemanticID{Encoder,Decoder}Module. Discover defensively.
        enc_stack = self._find_t5_stack(self.encoder.encoder)
        dec_stack = self._find_t5_stack(self.decoder.decoder)

        cfg = enc_stack.config
        act_fn = ACT2FN[getattr(cfg, "dense_act_fn", "relu")]
        common = dict(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout_rate,
            eps=cfg.layer_norm_epsilon,
            act_fn=act_fn,
            num_experts=self.num_experts,
            num_behaviors=self.num_behaviors,
            behavior_embedding_dim=behavior_embedding_dim,
        )

        se = set(sparse_layers_encoder or [])
        sd = set(sparse_layers_decoder or [])
        be = set(behavior_injection_encoder or [])
        bd = set(behavior_injection_decoder or [])

        # Swap the FFN (always the LAST sub-layer of a T5 block) in every encoder/decoder
        # block. Keep references so we can stash indices before each backbone call.
        self._encoder_ffns: List[PBAFeedForward] = []
        for i, blk in enumerate(enc_stack.block):
            ff = PBAFeedForward(
                is_sparse=(i in se), behavior_injection=(i in be), **common
            )
            blk.layer[-1] = ff
            self._encoder_ffns.append(ff)

        self._decoder_ffns: List[PBAFeedForward] = []
        for i, blk in enumerate(dec_stack.block):
            ff = PBAFeedForward(
                is_sparse=(i in sd), behavior_injection=(i in bd), **common
            )
            blk.layer[-1] = ff
            self._decoder_ffns.append(ff)

    @staticmethod
    def _find_t5_stack(module):
        """Return the T5Stack (the module owning `.block`) inside `module`.

        Handles both a bare T5Stack and a T5EncoderModel (whose stack is at `.encoder`).
        """
        if hasattr(module, "block"):
            return module
        if hasattr(module, "encoder") and hasattr(module.encoder, "block"):
            return module.encoder
        raise AttributeError(f"Could not locate a T5Stack (.block) inside {type(module)}")

    # ------------------------------------------------------------------
    # Position / behavior index construction (deterministic; = MBGen's routers)
    # ------------------------------------------------------------------
    def _item_position_pattern(self, device: torch.device) -> torch.Tensor:
        """Within-item expert ids for one item's (1+H) tokens.

        [behavior -> 1, sid0 -> 2, ..., sid_{H-1} -> 1+H]. With moe_behavior_only the SID
        levels collapse to a single shared expert (behavior -> 1, all SIDs -> 2)."""
        if self.moe_behavior_only:
            return torch.tensor(
                [1] + [2] * self.num_hierarchies, dtype=torch.long, device=device
            )
        return torch.arange(1, self.num_positions + 1, dtype=torch.long, device=device)

    def _encoder_indices(self, input_ids, attention_mask, has_user):
        """pos/beh indices for the encoder sequence [(user), items...] -> [B, L']."""
        B, L = input_ids.shape
        stride = 1 + self.num_hierarchies
        n_items = L // stride
        device = input_ids.device

        reps = (L + stride - 1) // stride
        pat = self._item_position_pattern(device).repeat(reps)[:L]  # [L]
        pos = pat.unsqueeze(0).expand(B, -1).clone()
        pos = pos * attention_mask.long()  # pad -> expert 0

        # behavior value per item = the behavior token at each item's slot 0.
        beh_tokens = input_ids[:, ::stride]  # [B, n_slots]
        beh = torch.zeros(B, L, dtype=torch.long, device=device)
        # SID slots (within-item offsets 1..H) carry behavior_id + 1; behavior slot stays 0.
        for h in range(1, stride):
            n_slots = beh[:, h::stride].shape[1]
            beh[:, h::stride] = beh_tokens[:, :n_slots] + 1
        beh = beh * attention_mask.long()

        if has_user:
            zc = torch.zeros(B, 1, dtype=torch.long, device=device)
            pos = torch.cat([zc, pos], dim=1)
            beh = torch.cat([zc, beh], dim=1)
        return pos, beh

    def _decoder_indices(self, future_ids, cache_valid, batch_size, device):
        """pos/beh indices for the decoder call, matching parent decoder_forward_pass.

        Decoder block layout is [BOS, behavior, sid0, ..., sid_{H-1}]:
          BOS -> expert 0 (beh 0), behavior -> expert 1 (beh 0), sid_k -> expert 2+k
          (beh = item behavior + 1).
        """
        pat = self._item_position_pattern(device)  # [1+H] experts for [behavior, sids...]

        if future_ids is None:
            # step 0: decoder sees only BOS.
            z = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            return z, z

        F = future_ids.shape[1]
        beh_val = future_ids[:, 0].long() + 1  # [batch] item behavior (+1 offset)

        if cache_valid:
            # single new token at within-block position F (BOS=0, behavior=1, sid_k=2+k).
            new_pos = pat[F - 1].view(1, 1).expand(batch_size, 1).clone()
            # position F: F==1 -> behavior token (beh 0); F>=2 -> SID token (beh_val).
            if F == 1:
                new_beh = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            else:
                new_beh = beh_val.view(batch_size, 1)
            return new_pos, new_beh

        # full pass: [BOS] + future_ids
        pos = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.long, device=device),
             pat[:F].unsqueeze(0).expand(batch_size, -1)],
            dim=1,
        )
        beh = torch.zeros(batch_size, F + 1, dtype=torch.long, device=device)
        # future col 0 = behavior token (beh 0); cols 1..H = SID tokens (beh_val).
        for j in range(1, F):
            beh[:, j + 1] = beh_val
        return pos, beh

    def _set_ffn_indices(self, ffns, pos, beh):
        for ff in ffns:
            ff.pos_index = pos
            ff.beh_index = beh

    # ------------------------------------------------------------------
    # Backbone call wrappers: stash indices, then run the inherited pass
    # ------------------------------------------------------------------
    def encoder_forward_pass(self, attention_mask, input_ids, user_id):
        has_user = user_id is not None and self.user_embedding is not None
        pos, beh = self._encoder_indices(input_ids, attention_mask, has_user)
        self._set_ffn_indices(self._encoder_ffns, pos, beh)
        return super().encoder_forward_pass(
            attention_mask=attention_mask, input_ids=input_ids, user_id=user_id
        )

    def decoder_forward_pass(
        self,
        attention_mask=None,
        future_ids=None,
        encoder_output=None,
        attention_mask_for_encoder=None,
        use_cache=False,
        past_key_values=None,
    ):
        batch_size = (
            future_ids.size(0) if future_ids is not None else encoder_output.size(0)
        )
        device = (
            future_ids.device if future_ids is not None else encoder_output.device
        )
        cache_valid = self._is_kv_cache_valid(kv_cache=past_key_values)
        pos, beh = self._decoder_indices(future_ids, cache_valid, batch_size, device)
        self._set_ffn_indices(self._decoder_ffns, pos, beh)
        return super().decoder_forward_pass(
            attention_mask=attention_mask,
            future_ids=future_ids,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )