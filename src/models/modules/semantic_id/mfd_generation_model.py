"""MFD (Modality-Factorized Decoding) multi-behavior generative recommender.

Structural companion to the §12g count-level result: given the shared/fusion code L0=f,
the text code L1=t and image code L2=i are near-independent (CMI(t;i|f) small vs H(t|f),
H(i|f)). MFD is the DECODER that spends that license — it drops the chain edge t -> i:

    chain      P(b,s|x) = P(b|x) P(f|x,b) P(t|x,b,f) P(i|x,b,f,t)
    factorized P(b,s|x) = P(b|x) P(f|x,b) P(t|x,b,f) P(i|x,b,f)      <- removed edge = the 1.7%

Everything EXCEPT the decoder factorization is inherited unchanged from
`SemanticIDMultiBehaviorEncoderDecoder`, so t1/t2/t3 metrics stay directly comparable to the
chain baselines (this isolates the decoder contribution):

  * Embeddings: the same concatenated [L0|L1|L2|behavior] table, sid_level_offsets /
    behavior_offset, and MD-RQ-VAE codebook_init injection. No new item/catalog embeddings.
  * Encoder: unchanged (flat interleaved [b,f,t,i,...], stride 4, use_item_pe still available).
  * Evaluator / focal / behavior_weighted_sid / data builders / DDP: unchanged.

The DECODER change (the only change):
  Teacher-forced decoder input per item is [BOS, behavior, f] — 3 positions, not 4.
      position  p0=BOS        p1=behavior      p2=shared code f  (= h_f)
      predicts  b @ p0        f @ p1           t AND i @ p2  (two heads on the SAME h_f)
  t and i are never fed as decoder inputs — they exist only as targets. Both modality heads
  (decoder_mlp[1] text, decoder_mlp[2] image) read the same hidden state h_f; this shared-
  state contention is the whole point. The data pipeline is untouched: the label is still the
  full [b,f,t,i] 4-tuple (t and i are needed as targets); MFD simply feeds the decoder the
  first two columns and reads all three output positions. So NextKTokenMasking stays next_k=4
  and the encoder stays stride 4 — only the teacher-forced decoder slice is 3 wide.

  Loss  L = CE_beh (focal if enabled) + CE_f + CE_t|f + CE_i|f      (one term re-parented)

Generation is a compositional beam with agreement scoring (§5):
  1. beam over (behavior, f): keep top-K_f (b,f) roots.
  2. for each root, one forward to h_f, read BOTH heads -> top-K_t text + top-K_i image.
  3. compose (f,t,i) in K_t x K_i, filter by the catalog constraint (the precomputed set of
     valid SID tuples in self.codebooks — replaces the chain's implicit validity-by-
     construction), and score each survivor
         score(f,t,i) = logP(f|.) + (1/tau) * [ logP(t|f,.) + logP(i|f,.) ]
     tau is the agreement temperature (frozen, inference-swept: tau>1 softens the
     independence assumption where the branches disagree, tau<1 sharpens mutual verification;
     tau=1 is exact product-of-experts). A learnable tau is intentionally omitted so the
     TRAINING path is byte-for-byte the chain baseline's — the ablation isolates the decoder.
  4. global top-K_pool across all roots -> the evaluator takes top-10 by score, as today.

Ablations this file supports via config:
  (a) chain vs factorized      : run the chain PBA/tiger baseline vs this class, same everything
  (b) agreement vs single-branch: mfd_score_text / mfd_score_image (drop one logP term)
  (c) tau sweep                : mfd_tau in {0.7, 1.0, 1.5}
  (d) K_t / K_i asymmetry       : mfd_top_k_text / mfd_top_k_image
"""

from __future__ import annotations

from typing import Optional, Tuple

from collections import defaultdict

import torch

from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDMultiBehaviorEncoderDecoder,
)


class MFDMultiBehaviorEncoderDecoder(SemanticIDMultiBehaviorEncoderDecoder):
    """Modality-Factorized-Decoding variant of the multi-behavior TIGER model.

    Requires num_hierarchies == 3 (shared / text / image). Mutually exclusive with use_bmtv.
    """

    def __init__(
        self,
        use_modality_factorized: bool = True,
        mfd_top_k_shared: int = 10,
        mfd_top_k_text: int = 10,
        mfd_top_k_image: int = 10,
        mfd_pool_size: int = 50,
        mfd_tau: float = 1.0,
        mfd_catalog_constraint: bool = True,
        mfd_score_text: bool = True,
        mfd_score_image: bool = True,
        mfd_behavior_marginalized: bool = True,
        mfd_behavior_weight: float = 1.0,
        mfd_audit_dump: bool = False,
        mfd_audit_path: str = "/scratch/yw8866/rec-tmall/mfd_audit_dump.pt",
        mfd_audit_max_events: int = 200000,
        mfd_audit_pv_cap: int = 100000,
        mfd_audit_pv_id: int = 0,
        mfd_dump_memory: bool = False,
        mfd_wide_catalog_pool: bool = False,
        use_pmq: bool = False,
        use_query_f: bool = False,
        decoder_slot_type_emb: bool = True,
        mfd_warmstart_path: Optional[str] = None,
        use_cmir: bool = False,
        cmir_num_rounds: int = 1,
        cmir_estimate_top_m: int = 3,
        cmir_loss_weight: float = 1.0,
        cmir_refine_top_m: int = -1,
        cmir_scheduled_sampling: bool = False,
        cmir_ss_p0: float = 0.25,
        cmir_ss_decay_steps: int = 20000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if self.num_hierarchies != 3:
            raise ValueError(
                f"MFD requires num_hierarchies == 3 (shared/text/image), got {self.num_hierarchies}"
            )
        if getattr(self, "use_bmtv", False):
            raise ValueError("use_modality_factorized and use_bmtv are mutually exclusive")
        if getattr(self, "use_semantic_align", False) or getattr(self, "use_semantic_rerank", False):
            raise ValueError(
                "use_semantic_align / use_semantic_rerank are chain-decoder features and are "
                "not supported under MFD (the item-summary hidden state layout differs). "
                "Leave them false for MFD runs."
            )
        if not (mfd_score_text or mfd_score_image):
            raise ValueError("at least one of mfd_score_text / mfd_score_image must be True")

        self.use_modality_factorized = use_modality_factorized
        self.mfd_top_k_shared = mfd_top_k_shared
        self.mfd_top_k_text = mfd_top_k_text
        self.mfd_top_k_image = mfd_top_k_image
        self.mfd_pool_size = mfd_pool_size
        self.mfd_tau = float(mfd_tau)
        self.mfd_catalog_constraint = mfd_catalog_constraint
        self.mfd_score_text = mfd_score_text
        self.mfd_score_image = mfd_score_image
        self.mfd_behavior_marginalized = mfd_behavior_marginalized
        self.mfd_behavior_weight = float(mfd_behavior_weight)

        # ---- audit dump (§0 audits: one instrumented test pass, three offline analyses) ----
        self.mfd_audit_dump = mfd_audit_dump
        self.mfd_audit_path = mfd_audit_path
        self.mfd_audit_max_events = mfd_audit_max_events
        self.mfd_audit_pv_cap = mfd_audit_pv_cap
        self.mfd_audit_pv_id = mfd_audit_pv_id
        # Stage-1 reranker candidate dump: when true, the audit dump ALSO stores the frozen
        # encoder memory (full history hidden states) + its mask per event, so the offline
        # listwise reranker (train_mfd_reranker.py) can cross-attend into it with no T5 loaded.
        # The per-behavior pools + 4 component log-probs the reranker needs are ALREADY dumped
        # by _audit_forward; this flag only adds the memory. Default False -> audit path is
        # byte-for-byte unchanged (baseline bit-identical when disabled).
        self.mfd_dump_memory = mfd_dump_memory
        # Wide-catalog pool (G5): when true, the audit pool is the top-K_pool by SID score over the
        # UNIQUE CATALOG TUPLES directly (equivalent to K_t=K_i=full head width, no t/i pruning),
        # scored in one vectorized pass -- avoids the infeasible dense K_f*K_t*K_i grid at 1024^2.
        # Only affects _audit_forward's pool construction; requires the catalog constraint on.
        self.mfd_wide_catalog_pool = mfd_wide_catalog_pool
        self._cat_ftii = None
        self._audit_cols = defaultdict(list)
        self._audit_n = 0
        self._audit_pv = 0

        # Radix for packing a (f,t,i) tuple into one int64 key. Use the per-level codebook
        # widths so keys never collide (text/image both 1024, shared 2048).
        self._k0 = int(self.sid_level_sizes[0])   # shared / f  head width
        self._k1 = int(self.sid_level_sizes[1])   # text  / t  head width
        self._k2 = int(self.sid_level_sizes[2])   # image / i  head width

        # ---- catalog constraint: the sorted set of valid (f,t,i) keys ----
        # self.codebooks is (N_items, 3) already ([:,0]=shared, [:,1]=text, [:,2]=image).
        # Non-persistent buffer: moves with the module (.to(device)) but is NOT written into
        # checkpoints (it is fully derivable from semantic_ids.pt). ~5.4M int64 ≈ 43MB.
        if mfd_catalog_constraint:
            cb = self.codebooks.long()
            keys = (cb[:, 0] * self._k1 + cb[:, 1]) * self._k2 + cb[:, 2]
            keys = torch.unique(keys)  # sorted, deduped
            self.register_buffer("_mfd_catalog_keys", keys, persistent=False)
        else:
            self._mfd_catalog_keys = None

        # ---- Query slots (PMQ / Query-f): learnable query positions in the decoder block ----
        # Per-item decoder block is a slot list. Ordering is load-bearing: q_f is BEFORE f so
        # causal masking already prevents q_f from seeing the teacher-forced f*; q_t/q_i are
        # AFTER f so they condition on it (P(t|f), P(i|f)). Encoder is UNTOUCHED (stride 4).
        #   both false  -> [BOS, beh, f]                (current MFD; uses the inherited path)
        #   pmq only    -> [BOS, beh, f, q_t, q_i]      (each modality its own hidden state)
        #   full        -> [BOS, beh, q_f, f, q_t, q_i] (shared head its own capacity too)
        self.use_pmq = use_pmq
        self.use_query_f = use_query_f
        self.decoder_slot_type_emb = decoder_slot_type_emb
        self._slot_mode = use_pmq or use_query_f

        slots = ["bos", "beh"]
        if use_query_f:
            slots.append("q_f")
        slots.append("f")
        if use_pmq:
            slots += ["q_t", "q_i"]
        self._slots = slots
        self._slot_pos = {n: i for i, n in enumerate(slots)}
        # head-read positions (which hidden state each head reads)
        self._pos_beh = self._slot_pos["bos"]                                    # behavior @ h_bos
        self._pos_f = self._slot_pos["q_f"] if use_query_f else self._slot_pos["beh"]
        # (t/i are read at the LAST slot of their own pass -> no fixed canonical index needed)

        d = self.embedding_dim
        if use_query_f:
            self.query_f = torch.nn.Parameter(torch.randn(d) * 0.02)
        if use_pmq:
            self.query_t = torch.nn.Parameter(torch.randn(d) * 0.02)
            self.query_i = torch.nn.Parameter(torch.randn(d) * 0.02)
        # Slot identity signal (T5 has no absolute PE, only relative bias). Zero-init -> a warm
        # start begins at parity with the current checkpoint (adds nothing until trained).
        if self._slot_mode and decoder_slot_type_emb:
            self.decoder_slot_type_embedding = torch.nn.Embedding(len(slots), d)
            torch.nn.init.zeros_(self.decoder_slot_type_embedding.weight)
        else:
            self.decoder_slot_type_embedding = None

        # ---- CMIR (Cross-Modal Iterative Refinement) ----
        # Round 0: q_t -> P0(t|f), q_i -> P0(i|f) (masked from each other, as today). From P0 we
        # build soft expected code embeddings (top-m, prob-weighted, stop-grad). Round 1 injects
        # the OTHER modality's estimate into the query slot: q_t += proj_i2t(i_est) -> P1(t|f,i_est),
        # q_i += proj_t2i(t_est) -> P1(i|f,t_est), still one pass each (mutually masked). proj_* are
        # ZERO-init so at step 0 Round 1 == Round 0 (query_add=0) and the Run-C warm start survives.
        self.use_cmir = use_cmir
        self.cmir_num_rounds = cmir_num_rounds
        self.cmir_estimate_top_m = cmir_estimate_top_m
        self.cmir_loss_weight = float(cmir_loss_weight)
        self.cmir_refine_top_m = cmir_refine_top_m
        self.cmir_scheduled_sampling = cmir_scheduled_sampling
        self.cmir_ss_p0 = float(cmir_ss_p0)
        self.cmir_ss_decay_steps = int(cmir_ss_decay_steps)
        if use_cmir:
            if not use_pmq:
                raise ValueError("use_cmir requires use_pmq (needs the separate q_t / q_i passes)")
            if cmir_num_rounds != 1:
                raise ValueError("this CMIR implementation supports exactly 1 refinement round")
            self.cmir_proj_i2t = torch.nn.Linear(d, d)  # image estimate -> text-query injection
            self.cmir_proj_t2i = torch.nn.Linear(d, d)  # text estimate  -> image-query injection
            for p in (self.cmir_proj_i2t, self.cmir_proj_t2i):
                torch.nn.init.zeros_(p.weight); torch.nn.init.zeros_(p.bias)

        # ---- Warm start: load a prior checkpoint's weights (strict=False so the new query
        # params / slot-type emb stay at init). Distinct from ckpt_path (which RESUMES). ----
        if mfd_warmstart_path:
            sd = torch.load(mfd_warmstart_path, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[MFD warmstart] {mfd_warmstart_path}: loaded (missing={len(missing)} "
                  f"unexpected={len(unexpected)}); new query/slot params kept at init.", flush=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _valid_tuple_mask(self, f: torch.Tensor, t: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Boolean mask (same shape as inputs) — is (f,t,i) a catalog SID tuple?"""
        if self._mfd_catalog_keys is None:
            return torch.ones_like(f, dtype=torch.bool)
        keys = self._mfd_catalog_keys
        cand = (f.long() * self._k1 + t.long()) * self._k2 + i.long()
        pos = torch.searchsorted(keys, cand)
        pos = pos.clamp(max=keys.numel() - 1)
        return keys[pos] == cand

    def _catalog_triples(self):
        """Unique catalog SID tuples as (CF, CT, CI), each (M_unique,) long on the module device.
        Unpacked from the packed catalog keys; cached. Used by the wide-catalog pool path."""
        if self._cat_ftii is None:
            keys = self._mfd_catalog_keys
            if keys is None:
                cb = self.codebooks.long()
                keys = torch.unique((cb[:, 0] * self._k1 + cb[:, 1]) * self._k2 + cb[:, 2])
            ci = keys % self._k2
            ct = (keys // self._k2) % self._k1
            cf = keys // (self._k1 * self._k2)
            self._cat_ftii = (cf.long(), ct.long(), ci.long())
        return self._cat_ftii

    # ------------------------------------------------------------------
    # Slot-mode (PMQ / Query-f) decoder plumbing. Single target-item block; the encoder holds
    # the history and is untouched. Cache-free (blocks are <=5 slots).
    #
    # ROBUSTNESS CHOICE: q_t and q_i are decoded in SEPARATE passes (each the last slot of its
    # own block) rather than one block with a custom q_i-/->q_t mask. Since the two never coexist
    # in a pass, they are conditionally independent given f BY CONSTRUCTION (exact P(t|f)·P(i|f)),
    # and every pass uses only STANDARD CAUSAL masking (attention_mask=None -> T5's own causal),
    # the same path the current MFD relies on. Cost: one extra tiny modality forward under PMQ.
    # ------------------------------------------------------------------
    def _slot_names(self, include_f, extra):
        """Canonical slot-name list for a pass. include_f adds the f slot; extra in
        {None,'q_t','q_i'} appends a modality query as the last (read) slot."""
        names = ["bos", "beh"]
        if self.use_query_f:
            names.append("q_f")
        if include_f:
            names.append("f")
        if extra is not None:
            names.append(extra)
        return names

    def _slot_embed(self, name, b, f, M, dev):
        table = self.get_embedding_table("decoder")
        if name == "bos":
            e = self.decoder.bos_token.expand(M, -1)
        elif name == "beh":
            e = table(b.long() + self.behavior_offset)
        elif name == "q_f":
            e = self.query_f.unsqueeze(0).expand(M, -1)
        elif name == "f":
            e = table(f.long() + self.sid_level_offsets[0])
        elif name == "q_t":
            e = self.query_t.unsqueeze(0).expand(M, -1)
        elif name == "q_i":
            e = self.query_i.unsqueeze(0).expand(M, -1)
        if self.decoder_slot_type_embedding is not None:
            # index by the slot's CANONICAL position so identity is stable across passes
            tid = torch.tensor(self._slot_pos[name], device=dev)
            e = e + self.decoder_slot_type_embedding(tid).unsqueeze(0)
        return e

    def _slot_run(self, names, b, f, enc, enc_mask, query_add=None):
        """Build the block from `names` and run the decoder with STANDARD causal masking.
        `query_add` (M, d), if given, is added to the LAST slot's input embedding (CMIR uses this
        to inject the cross-modal estimate into the modality query). Returns (M, len(names), d)."""
        M = b.shape[0]; dev = b.device
        slots = [self._slot_embed(n, b, f, M, dev) for n in names]
        if query_add is not None:
            slots[-1] = slots[-1] + query_add   # inject into the modality query BEFORE stacking
        emb = torch.stack(slots, dim=1)
        return self.decoder(
            sequence_embedding=emb,
            attention_mask=None,               # None -> T5 applies its own causal mask
            encoder_output=enc,
            encoder_attention_mask=enc_mask,
            use_cache=False,
            past_key_values=None,
        )  # (M, len(names), d)

    def _slot_f_logp(self, enc, enc_mask, b_ids):
        """Stage A: block [BOS, beh, (q_f)] (no f slot) -> f logits @ last slot, behavior @ BOS.
        b_ids: (M,). Returns f_logits (M,k0), beh_logits (M,nb)."""
        names = self._slot_names(include_f=False, extra=None)
        h = self._slot_run(names, b_ids, None, enc, enc_mask)
        return self.decoder.decoder_mlp[0](h[:, -1]), self.behavior_mlp(h[:, 0])

    def _slot_modality(self, enc, enc_mask, b_sel, f_sel):
        """Stage B: log P(t|f), log P(i|f). PMQ -> two passes ([...,f,q_t] and [...,f,q_i]),
        else one pass ([...,f]) whose h_f feeds both heads. Returns (B,Kf,k1),(B,Kf,k2)."""
        B, Kf = f_sel.shape
        b = b_sel.reshape(B * Kf); f = f_sel.reshape(B * Kf)
        enc_rep = enc.repeat_interleave(Kf, 0); mask_rep = enc_mask.repeat_interleave(Kf, 0)
        if self.use_pmq:
            h_t = self._slot_run(self._slot_names(True, "q_t"), b, f, enc_rep, mask_rep)[:, -1]
            h_i = self._slot_run(self._slot_names(True, "q_i"), b, f, enc_rep, mask_rep)[:, -1]
        else:
            h = self._slot_run(self._slot_names(True, None), b, f, enc_rep, mask_rep)[:, -1]
            h_t = h_i = h
        t_logp = torch.log_softmax(self.decoder.decoder_mlp[1](h_t), dim=-1)
        i_logp = torch.log_softmax(self.decoder.decoder_mlp[2](h_i), dim=-1)
        return t_logp.reshape(B, Kf, -1), i_logp.reshape(B, Kf, -1)

    def _stageA_all_behaviors(self, enc, enc_mask, B):
        """Behavior log-probs (B,nb) + f log-probs under every behavior (B,nb,k0)."""
        nb = self.num_behaviors; dev = enc.device
        beh_ids = torch.arange(nb, device=dev)
        if self._slot_mode:
            b_rep = beh_ids.view(1, nb).expand(B, nb).reshape(B * nb)
            f_logits, beh_logits = self._slot_f_logp(
                enc.repeat_interleave(nb, 0), enc_mask.repeat_interleave(nb, 0), b_rep
            )
            f_logp = torch.log_softmax(f_logits, dim=-1).reshape(B, nb, self._k0)
            # h_bos is identical across the behavior branches (BOS precedes the beh slot) -> take branch 0
            beh_logp = torch.log_softmax(beh_logits.reshape(B, nb, -1)[:, 0], dim=-1)
            return beh_logp, f_logp
        dec0 = self.decoder_forward_pass(
            future_ids=None, encoder_output=enc, attention_mask_for_encoder=enc_mask, use_cache=False
        )
        beh_logp = torch.log_softmax(self.behavior_mlp(dec0[:, -1, :]), dim=-1)
        fut_b = beh_ids.view(1, nb, 1).expand(B, nb, 1).reshape(B * nb, 1)
        dec1 = self.decoder_forward_pass(
            future_ids=fut_b, encoder_output=enc.repeat_interleave(nb, 0),
            attention_mask_for_encoder=enc_mask.repeat_interleave(nb, 0), use_cache=False,
        )
        f_logp = torch.log_softmax(
            self.decoder.decoder_mlp[0](dec1[:, -1, :]), dim=-1
        ).reshape(B, nb, self._k0)
        return beh_logp, f_logp

    # ------------------------------------------------------------------
    # CMIR (Cross-Modal Iterative Refinement)
    # ------------------------------------------------------------------
    def _soft_estimate(self, logp, level, gt=None, ss_mask=None):
        """Soft expected code embedding from a modality log-prob (top-m, prob-weighted), at the
        given SID level (1=text, 2=image). Stop-gradient by default. If scheduled sampling is
        active (ss_mask), substitute the GT code embedding where masked. Returns (M, d)."""
        p = logp.exp()
        m = min(self.cmir_estimate_top_m, p.size(-1))
        vals, idx = p.topk(m, dim=-1)                                  # (M, m)
        w = vals / vals.sum(-1, keepdim=True).clamp(min=1e-9)          # renormalise over top-m
        table = self.get_embedding_table("decoder")
        off = self.sid_level_offsets[level]
        est = (w.unsqueeze(-1) * table(idx + off)).sum(1).detach()     # (M, d), stop-grad
        if gt is not None and ss_mask is not None and bool(ss_mask.any()):
            gt_emb = table(gt.long() + off).detach()
            est = torch.where(ss_mask.unsqueeze(-1), gt_emb, est)
        return est

    def _cmir_modality(self, enc, enc_mask, b_sel, f_sel, score_root=None):
        """Inference Stage B under CMIR: Round 0 -> estimates -> Round 1. Returns the ROUND-1
        (B,Kf,k1)/(B,Kf,k2) log-probs used for composition/scoring."""
        B, Kf = f_sel.shape
        b = b_sel.reshape(-1); f = f_sel.reshape(-1)
        enc_rep = enc.repeat_interleave(Kf, 0); mask_rep = enc_mask.repeat_interleave(Kf, 0)
        nt = self._slot_names(True, "q_t"); ni = self._slot_names(True, "q_i")
        # Round 0 (masked from each other)
        h_t0 = self._slot_run(nt, b, f, enc_rep, mask_rep)[:, -1]
        h_i0 = self._slot_run(ni, b, f, enc_rep, mask_rep)[:, -1]
        t_lp0 = torch.log_softmax(self.decoder.decoder_mlp[1](h_t0), -1)
        i_lp0 = torch.log_softmax(self.decoder.decoder_mlp[2](h_i0), -1)
        i_est = self._soft_estimate(i_lp0, 2)   # image estimate -> conditions q_t
        t_est = self._soft_estimate(t_lp0, 1)   # text estimate  -> conditions q_i
        # Round 1 (each still one pass, mutually masked; only cross-round info flows)
        h_t1 = self._slot_run(nt, b, f, enc_rep, mask_rep, query_add=self.cmir_proj_i2t(i_est))[:, -1]
        h_i1 = self._slot_run(ni, b, f, enc_rep, mask_rep, query_add=self.cmir_proj_t2i(t_est))[:, -1]
        t_lp1 = torch.log_softmax(self.decoder.decoder_mlp[1](h_t1), -1)
        i_lp1 = torch.log_softmax(self.decoder.decoder_mlp[2](h_i1), -1)
        # optional cost cap: only the top-M f branches (by Round-0 score) keep Round 1
        M = self.cmir_refine_top_m
        if score_root is not None and 0 < M < Kf:
            sel = score_root.topk(M, dim=-1).indices
            refine = torch.zeros(B, Kf, dtype=torch.bool, device=f.device)
            refine.scatter_(1, sel, True)
            refine = refine.reshape(-1, 1)
            t_lp1 = torch.where(refine, t_lp1, t_lp0)
            i_lp1 = torch.where(refine, i_lp1, i_lp0)
        return t_lp1.reshape(B, Kf, -1), i_lp1.reshape(B, Kf, -1)

    def _modality_heads(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        b_sel: torch.Tensor,   # (B, Kf)
        f_sel: torch.Tensor,   # (B, Kf)
        score_root: torch.Tensor = None,   # (B, Kf) for the CMIR top-M-branch cap; None -> refine all
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One cache-free forward per (b,f) root -> (text_logp, img_logp), (B,Kf,k1)/(B,Kf,k2).
        CMIR -> Round-1 two-round refinement; slot mode -> query-slot block; else plain [BOS,b,f]."""
        if self.use_cmir:
            return self._cmir_modality(encoder_output, encoder_mask, b_sel, f_sel, score_root)
        if self._slot_mode:
            return self._slot_modality(encoder_output, encoder_mask, b_sel, f_sel)
        B, Kf = f_sel.shape
        fut_bf = torch.stack([b_sel, f_sel], dim=-1).reshape(B * Kf, 2)  # (B*Kf, 2)
        enc_rep = encoder_output.repeat_interleave(Kf, dim=0)
        mask_rep = encoder_mask.repeat_interleave(Kf, dim=0)
        dec = self.decoder_forward_pass(
            future_ids=fut_bf,
            encoder_output=enc_rep,
            attention_mask_for_encoder=mask_rep,
            use_cache=False,
        )  # (B*Kf, 3, d)
        h_f = dec[:, -1, :]  # after f -> the shared item-summary hidden state
        t_logp = torch.log_softmax(self.decoder.decoder_mlp[1](h_f), dim=-1)  # (B*Kf, k1)
        i_logp = torch.log_softmax(self.decoder.decoder_mlp[2](h_f), dim=-1)  # (B*Kf, k2)
        return t_logp.reshape(B, Kf, -1), i_logp.reshape(B, Kf, -1)

    def _compose_and_pool(
        self,
        b_sel: torch.Tensor,     # (B, Kf)
        f_sel: torch.Tensor,     # (B, Kf)
        score_root: torch.Tensor,  # (B, Kf)  logP(b)+logP(f|b)  (t2: logP(f|b_gt))
        t_logp: torch.Tensor,    # (B, Kf, k1)
        i_logp: torch.Tensor,    # (B, Kf, k2)
    ):
        """Compose top-Kt x top-Ki modality candidates per root, catalog-filter, score, and
        return the global top-K_pool (b, f, t, i) tuples + their scores per example."""
        B, Kf = f_sel.shape
        Kt = min(self.mfd_top_k_text, t_logp.size(-1))
        Ki = min(self.mfd_top_k_image, i_logp.size(-1))
        inv_tau = 1.0 / self.mfd_tau

        tv, ti = t_logp.topk(Kt, dim=-1)   # (B, Kf, Kt)
        iv, ii = i_logp.topk(Ki, dim=-1)   # (B, Kf, Ki)

        # additive log-scores over the Kt x Ki grid per root
        w_t = 1.0 if self.mfd_score_text else 0.0
        w_i = 1.0 if self.mfd_score_image else 0.0
        comp = (
            score_root.view(B, Kf, 1, 1)
            + inv_tau * w_t * tv.view(B, Kf, Kt, 1)
            + inv_tau * w_i * iv.view(B, Kf, 1, Ki)
        )  # (B, Kf, Kt, Ki)

        b_cand = b_sel.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki)
        f_cand = f_sel.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki)
        t_cand = ti.view(B, Kf, Kt, 1).expand(B, Kf, Kt, Ki)
        i_cand = ii.view(B, Kf, 1, Ki).expand(B, Kf, Kt, Ki)

        M = Kf * Kt * Ki
        comp = comp.reshape(B, M)
        b_cand = b_cand.reshape(B, M)
        f_cand = f_cand.reshape(B, M)
        t_cand = t_cand.reshape(B, M)
        i_cand = i_cand.reshape(B, M)

        # catalog constraint: invalid compositions are discarded (score -> -inf)
        if self.mfd_catalog_constraint:
            valid = self._valid_tuple_mask(f_cand, t_cand, i_cand)
            comp = comp.masked_fill(~valid, float("-inf"))

        Kpool = min(self.mfd_pool_size, M)
        score_pool, pidx = comp.topk(Kpool, dim=-1)  # (B, Kpool)
        gather = lambda x: torch.gather(x, 1, pidx)
        return gather(b_cand), gather(f_cand), gather(t_cand), gather(i_cand), score_pool

    def _cmir_model_step(self, model_input, fut_ids):
        """Teacher-forced CMIR training. Round 0 gives beh/f/t0/i0 and the model's OWN estimates;
        Round 1 conditions each modality query on the other's estimate (stop-grad, optional
        scheduled sampling to GT). Loss = CE_beh + CE_f + CE_t0 + CE_i0 + lambda*(CE_t1 + CE_i1)."""
        inputs = {
            self.feature_to_model_input_map.get(k, k): v
            for k, v in model_input.transformed_sequences.items()
        }
        enc, enc_mask = self.encoder_forward_pass(
            attention_mask=model_input.mask, input_ids=inputs["input_ids"], user_id=inputs.get("user_id")
        )
        b = fut_ids[:, 0].long(); f = fut_ids[:, 1].long()
        t = fut_ids[:, 2].long(); i = fut_ids[:, 3].long()
        nt = self._slot_names(True, "q_t"); ni = self._slot_names(True, "q_i")
        # ---- Round 0 ----
        h_t0 = self._slot_run(nt, b, f, enc, enc_mask)
        h_i0 = self._slot_run(ni, b, f, enc, enc_mask)
        beh_logits = self.behavior_mlp(h_t0[:, self._pos_beh])
        f_logits = self.decoder.decoder_mlp[0](h_t0[:, self._pos_f])
        t0_logits = self.decoder.decoder_mlp[1](h_t0[:, -1])
        i0_logits = self.decoder.decoder_mlp[2](h_i0[:, -1])
        # ---- estimates (own Round-0 predictions; optional scheduled sampling to GT) ----
        ss = None
        if self.cmir_scheduled_sampling and self.training:
            p = self.cmir_ss_p0 * max(0.0, 1.0 - self.global_step / max(1, self.cmir_ss_decay_steps))
            ss = torch.rand(b.shape[0], device=b.device) < p
        i_est = self._soft_estimate(torch.log_softmax(i0_logits, -1), 2, gt=i, ss_mask=ss)
        t_est = self._soft_estimate(torch.log_softmax(t0_logits, -1), 1, gt=t, ss_mask=ss)
        # ---- Round 1 ----
        h_t1 = self._slot_run(nt, b, f, enc, enc_mask, query_add=self.cmir_proj_i2t(i_est))
        h_i1 = self._slot_run(ni, b, f, enc, enc_mask, query_add=self.cmir_proj_t2i(t_est))
        t1_logits = self.decoder.decoder_mlp[1](h_t1[:, -1])
        i1_logits = self.decoder.decoder_mlp[2](h_i1[:, -1])
        # ---- loss ----
        beh_loss_fn = self.behavior_loss_function or self.loss_function
        loss = beh_loss_fn(input=beh_logits, target=b)
        lw = self.cmir_loss_weight
        if self.behavior_weighted_sid:
            w = self.sid_behavior_weights.to(b.device)[b]; denom = w.sum().clamp(min=1e-8)
            ce = lambda lg, tg: (
                w * torch.nn.functional.cross_entropy(lg, tg, reduction="none")
            ).sum() / denom
            loss = (loss + ce(f_logits, f) + ce(t0_logits, t) + ce(i0_logits, i)
                    + lw * (ce(t1_logits, t) + ce(i1_logits, i)))
        else:
            L = self.loss_function
            loss = (loss + L(input=f_logits, target=f)
                    + L(input=t0_logits, target=t) + L(input=i0_logits, target=i)
                    + lw * (L(input=t1_logits, target=t) + L(input=i1_logits, target=i)))
        return h_t1, loss

    # ------------------------------------------------------------------
    # Training: teacher-forced 3-position decoder, two modality heads on h_f
    # ------------------------------------------------------------------
    def _slot_model_step(self, model_input, fut_ids):
        """Teacher-forced training for the query-slot decoder. Full block [BOS, beh, (q_f), f,
        (q_t), (q_i)] with f fed at the f slot; each head reads its own slot's hidden state.
        Preserves the MFD factorization P(f|hist,b)·P(t|hist,b,f)·P(i|hist,b,f)."""
        inputs = {
            self.feature_to_model_input_map.get(k, k): v
            for k, v in model_input.transformed_sequences.items()
        }
        enc, enc_mask = self.encoder_forward_pass(
            attention_mask=model_input.mask, input_ids=inputs["input_ids"], user_id=inputs.get("user_id")
        )
        b = fut_ids[:, 0].long(); f = fut_ids[:, 1].long()
        t = fut_ids[:, 2].long(); i = fut_ids[:, 3].long()
        # f-head reads q_f (or beh) at its canonical index; the block prefix preserves those
        # indices, and the modality head reads the LAST slot (q_t/q_i, or f when not PMQ).
        if self.use_pmq:
            h = self._slot_run(self._slot_names(True, "q_t"), b, f, enc, enc_mask)
            h_i = self._slot_run(self._slot_names(True, "q_i"), b, f, enc, enc_mask)
            beh_logits = self.behavior_mlp(h[:, self._pos_beh])
            f_logits = self.decoder.decoder_mlp[0](h[:, self._pos_f])
            t_logits = self.decoder.decoder_mlp[1](h[:, -1])       # q_t pass
            i_logits = self.decoder.decoder_mlp[2](h_i[:, -1])     # q_i pass
        else:
            h = self._slot_run(self._slot_names(True, None), b, f, enc, enc_mask)
            beh_logits = self.behavior_mlp(h[:, self._pos_beh])
            f_logits = self.decoder.decoder_mlp[0](h[:, self._pos_f])
            t_logits = self.decoder.decoder_mlp[1](h[:, -1])  # h_f
            i_logits = self.decoder.decoder_mlp[2](h[:, -1])
        beh_loss_fn = self.behavior_loss_function or self.loss_function
        loss = beh_loss_fn(input=beh_logits, target=b)
        if self.behavior_weighted_sid:
            w = self.sid_behavior_weights.to(b.device)[b]; denom = w.sum().clamp(min=1e-8)
            ce = lambda lg, tg: (
                w * torch.nn.functional.cross_entropy(lg, tg, reduction="none")
            ).sum() / denom
            loss = loss + ce(f_logits, f) + ce(t_logits, t) + ce(i_logits, i)
        else:
            loss = (loss + self.loss_function(input=f_logits, target=f)
                    + self.loss_function(input=t_logits, target=t)
                    + self.loss_function(input=i_logits, target=i))
        return h, loss

    def model_step(self, model_input, label_data=None):
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
            fut_ids = label_data.labels[label].reshape(model_input.mask.size(0), -1)
        # fut_ids: (B, 1+H) = [behavior, f, t, i]

        if self.use_cmir:
            return self._cmir_model_step(model_input, fut_ids)
        if self._slot_mode:
            return self._slot_model_step(model_input, fut_ids)

        # Decoder INPUT is only [behavior, f]; BOS is prepended inside decoder_forward_pass,
        # giving decoder positions [BOS, b, f] -> output (B, 3, d). Read ALL three positions
        # (do NOT drop the last, unlike the chain model): p0->behavior, p1->f, p2=h_f->t & i.
        dec_in = fut_ids[:, :2]
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=dec_in,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )  # (B, 3, d)

        h_bos, h_b, h_f = model_output[:, 0], model_output[:, 1], model_output[:, 2]

        # behavior loss @ p0 (focal if enabled)
        behavior_target = fut_ids[:, 0].long()
        behavior_loss_fn = self.behavior_loss_function or self.loss_function
        loss = behavior_loss_fn(input=self.behavior_mlp(h_bos), target=behavior_target)

        # SID losses: f @ p1 (from h_b), t & i @ p2 (both from the SAME h_f)
        f_logits = self.decoder.decoder_mlp[0](h_b)
        t_logits = self.decoder.decoder_mlp[1](h_f)
        i_logits = self.decoder.decoder_mlp[2](h_f)
        f_tgt = fut_ids[:, 1].long()
        t_tgt = fut_ids[:, 2].long()
        i_tgt = fut_ids[:, 3].long()

        if self.behavior_weighted_sid:
            w = self.sid_behavior_weights.to(behavior_target.device)[behavior_target]  # (B,)
            denom = w.sum().clamp(min=1e-8)
            ce = lambda logit, tgt: (
                w * torch.nn.functional.cross_entropy(logit, tgt, reduction="none")
            ).sum() / denom
            loss = loss + ce(f_logits, f_tgt) + ce(t_logits, t_tgt) + ce(i_logits, i_tgt)
        else:
            loss = (
                loss
                + self.loss_function(input=f_logits, target=f_tgt)
                + self.loss_function(input=t_logits, target=t_tgt)
                + self.loss_function(input=i_logits, target=i_tgt)
            )

        return model_output, loss

    # ------------------------------------------------------------------
    # Free generation (t1 / t3): compositional beam with agreement scoring
    # ------------------------------------------------------------------
    def generate_multibehavior(self, attention_mask, input_ids, user_id=None):
        encoder_output, encoder_mask = self.encoder_forward_pass(
            attention_mask=attention_mask, input_ids=input_ids, user_id=user_id
        )
        B = input_ids.size(0)
        device = encoder_output.device
        nb = self.num_behaviors

        # ---- Stage A: behavior log-probs (B,nb) + f log-probs under every behavior (B,nb,k0) ----
        # (routes to the query-slot decoder when use_pmq/use_query_f, else the plain [BOS,b,f] path)
        beh_logp, f_logp = self._stageA_all_behaviors(encoder_output, encoder_mask, B)

        # ---- roots: pick (b, f) roots and their w_b*logP(b)+logP(f|b) score ----
        w_b = self.mfd_behavior_weight
        if self.mfd_behavior_marginalized:
            # Behavior-marginalized pooling (the stronger version): DO NOT commit one behavior
            # in the beam. Give EACH of the nb behaviors its OWN top-Kf f-roots, so the per-
            # behavior SID retrieval that makes t2 strong runs for every behavior; then rank the
            # union of all nb pools by w_b*logP(b) + SID_score(.|b). Guarantees every behavior a
            # slice of the beam instead of letting one behavior crowd out the shared top-Kf.
            # Cost: nb x the f-branch modality forward (heads are cheap).
            kf = min(self.mfd_top_k_shared, self._k0)
            sc_f, f_idx = f_logp.topk(kf, dim=-1)                # (B, nb, kf)  logP(f|b)
            score_root = w_b * beh_logp.unsqueeze(-1) + sc_f     # (B, nb, kf)
            b_sel = (
                torch.arange(nb, device=device).view(1, nb, 1).expand(B, nb, kf)
            )                                                    # (B, nb, kf)
            f_sel = f_idx                                        # (B, nb, kf)
            Kf_tot = nb * kf
            b_sel = b_sel.reshape(B, Kf_tot)
            f_sel = f_sel.reshape(B, Kf_tot)
            score_root = score_root.reshape(B, Kf_tot)
        else:
            # Behavior-committed pooling: one shared beam over (b,f) -> global top-Kf roots.
            kf = min(self.mfd_top_k_shared, self._k0 * nb)
            joint = w_b * beh_logp.unsqueeze(-1) + f_logp        # (B, nb, k0)
            score_root, idx = joint.reshape(B, nb * self._k0).topk(kf, dim=-1)  # (B, kf)
            b_sel = idx // self._k0
            f_sel = idx % self._k0

        # ---- steps 2/2': both modality heads on h_f, compose, filter, pool ----
        t_logp, i_logp = self._modality_heads(encoder_output, encoder_mask, b_sel, f_sel, score_root)
        b, f, t, i, score = self._compose_and_pool(b_sel, f_sel, score_root, t_logp, i_logp)
        generated_ids = torch.stack([b, f, t, i], dim=-1)  # (B, Kpool, 4)
        return generated_ids, score  # (B, Kpool, 1+H), (B, Kpool)

    # ------------------------------------------------------------------
    # Conditioned generation (t2): behavior forced to GT, SID-only output
    # ------------------------------------------------------------------
    def generate_conditioned(self, attention_mask, input_ids, gt_behavior, user_id=None):
        encoder_output, encoder_mask = self.encoder_forward_pass(
            attention_mask=attention_mask, input_ids=input_ids, user_id=user_id
        )
        B = input_ids.size(0)
        Kf = min(self.mfd_top_k_shared, self._k0)

        # f log-probs conditioned on the GT behavior (slot decoder when in slot mode)
        if self._slot_mode:
            f_logits, _ = self._slot_f_logp(encoder_output, encoder_mask, gt_behavior.long())
            f_logp = torch.log_softmax(f_logits, dim=-1)  # (B, k0)
        else:
            dec1 = self.decoder_forward_pass(
                future_ids=gt_behavior.long().view(B, 1),
                encoder_output=encoder_output,
                attention_mask_for_encoder=encoder_mask,
                use_cache=False,
            )  # (B, 2, d)
            f_logp = torch.log_softmax(self.decoder.decoder_mlp[0](dec1[:, -1, :]), dim=-1)  # (B, k0)
        score_root, f_sel = f_logp.topk(Kf, dim=-1)  # (B, Kf)
        b_sel = gt_behavior.long().view(B, 1).expand(B, Kf)

        t_logp, i_logp = self._modality_heads(encoder_output, encoder_mask, b_sel, f_sel, score_root)
        _, f, t, i, score = self._compose_and_pool(b_sel, f_sel, score_root, t_logp, i_logp)
        generated_ids = torch.stack([f, t, i], dim=-1)  # (B, Kpool, H)  SIDs only
        return generated_ids, score

    # ==================================================================
    # AUDIT DUMP (mfd_audit_dump=true): one instrumented test pass captures, per event,
    # everything the three §0 audits need. Nothing is recomputed later.
    #   0a pool routing : per-behavior pools (SID_score sans behavior term) + beh_logp -> any
    #                     w_b union/routed re-ranking offline (committed is the cited baseline).
    #   0b contention   : teacher-forced GT-f branch reads (rank of t*,i* in the head softmax).
    #   0c coverage/rank: f* full rank (GT behavior), t*/i* ranks at GT f, and the pools.
    # ==================================================================
    @torch.no_grad()
    def _audit_forward(self, mask, input_ids, user_id, labels):
        enc, emask = self.encoder_forward_pass(
            attention_mask=mask, input_ids=input_ids, user_id=user_id
        )
        B = input_ids.size(0)
        dev = enc.device
        nb = self.num_behaviors
        ar = torch.arange(B, device=dev)
        gt_b = labels[:, 0].long(); gt_f = labels[:, 1].long()
        gt_t = labels[:, 2].long(); gt_i = labels[:, 3].long()

        # DIRECT catalog-validity of the GT tuple (the unit test): GT is a real tokenized item,
        # so it MUST be in the catalog. If any GT is absent, the constraint hash is wrong. This
        # separates a true catalog bug from mere pool truncation (composed+valid but rank>Kpool).
        gt_in_catalog = self._valid_tuple_mask(gt_f, gt_t, gt_i)  # (B,) bool

        # behavior + f log-probs under every behavior (slot-aware Stage A)
        beh_logp, f_logp = self._stageA_all_behaviors(enc, emask, B)  # (B,nb), (B,nb,k0)

        # f* rank within the GT-behavior shared softmax (full rank, for 0c near-misses)
        f_logp_gt = f_logp[ar, gt_b]                       # (B, k0)
        f_star_logp = f_logp_gt[ar, gt_f]                  # (B,)
        f_star_rank = (f_logp_gt > f_star_logp.unsqueeze(1)).sum(1)  # (B,) 0-based

        # teacher-forced GT-f branch reads (t & i read at their slots given the GT f) -> 0b/0c.
        # Reuse _modality_heads (slot-aware) with a single (b*, f*) root.
        t_lp1, i_lp1 = self._modality_heads(enc, emask, gt_b.view(B, 1), gt_f.view(B, 1))
        t_lp_gt = t_lp1[:, 0, :]  # (B, k1)  already log-softmax
        i_lp_gt = i_lp1[:, 0, :]  # (B, k2)
        tf_t_logp = t_lp_gt[ar, gt_t]; tf_t_rank = (t_lp_gt > tf_t_logp.unsqueeze(1)).sum(1)
        tf_i_logp = i_lp_gt[ar, gt_i]; tf_i_rank = (i_lp_gt > tf_i_logp.unsqueeze(1)).sum(1)

        # per-behavior pools (top-Kpool per behavior, catalog-filtered). SID_score EXCLUDES the
        # behavior term so 0a can add w_b*logP(b) for ANY w_b offline.
        Kf = min(self.mfd_top_k_shared, self._k0)
        Kt = min(self.mfd_top_k_text, self._k1)
        Ki = min(self.mfd_top_k_image, self._k2)
        Kp = self.mfd_pool_size
        inv_tau = 1.0 / self.mfd_tau
        w_t = 1.0 if self.mfd_score_text else 0.0
        w_i = 1.0 if self.mfd_score_image else 0.0
        pf = torch.zeros(B, nb, Kp, dtype=torch.long, device=dev)
        pt = torch.zeros_like(pf); pi = torch.zeros_like(pf)
        psc = torch.full((B, nb, Kp), float("-inf"), device=dev)
        plpf = torch.zeros(B, nb, Kp, device=dev)
        plpt = torch.zeros_like(plpf); plpi = torch.zeros_like(plpf)
        for b in range(nb):
            sc_f, f_idx = f_logp[:, b, :].topk(Kf, dim=-1)        # (B, Kf)
            b_sel = torch.full((B, Kf), b, dtype=torch.long, device=dev)
            t_lp, i_lp = self._modality_heads(enc, emask, b_sel, f_idx)  # (B,Kf,k1),(B,Kf,k2)

            # ---- G5 wide-catalog pool: score the unique catalog tuples directly (= K_t=K_i=full,
            # no dense grid). Per event, map each catalog tuple's shared code to its top-Kf root (or
            # drop it), gather the root's per-modality log-probs at that tuple's t/i, score, top-Kp.
            if self.mfd_wide_catalog_pool:
                CF, CT, CI = self._catalog_triples()              # (Mc,) each
                Mc = CF.numel()
                code2root = torch.full((B, self._k0), -1, dtype=torch.long, device=dev)
                code2root.scatter_(1, f_idx, torch.arange(Kf, device=dev).view(1, Kf).expand(B, Kf))
                root = code2root[:, CF]                            # (B, Mc) root index or -1
                valid = root >= 0
                rc = root.clamp(min=0)
                scf_i = torch.gather(sc_f, 1, rc)                 # (B, Mc)  logP(f|b)
                t_at = torch.gather(t_lp.reshape(B, Kf * self._k1), 1, rc * self._k1 + CT.view(1, Mc))
                i_at = torch.gather(i_lp.reshape(B, Kf * self._k2), 1, rc * self._k2 + CI.view(1, Mc))
                score = scf_i + inv_tau * (w_t * t_at + w_i * i_at)
                score = score.masked_fill(~valid, float("-inf"))
                kk = min(Kp, Mc)
                sc_pool, pidx = score.topk(kk, dim=-1)            # (B, kk)
                pf[:, b, :kk] = CF[pidx]; pt[:, b, :kk] = CT[pidx]; pi[:, b, :kk] = CI[pidx]
                psc[:, b, :kk] = sc_pool
                plpf[:, b, :kk] = torch.gather(scf_i, 1, pidx)
                plpt[:, b, :kk] = torch.gather(t_at, 1, pidx)
                plpi[:, b, :kk] = torch.gather(i_at, 1, pidx)
                continue

            tv, ti = t_lp.topk(Kt, dim=-1)                         # (B, Kf, Kt)
            iv, ii = i_lp.topk(Ki, dim=-1)                         # (B, Kf, Ki)
            lpf = sc_f.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki)
            comp = lpf + inv_tau * (w_t * tv.view(B, Kf, Kt, 1) + w_i * iv.view(B, Kf, 1, Ki))
            f_c = f_idx.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki)
            t_c = ti.view(B, Kf, Kt, 1).expand(B, Kf, Kt, Ki)
            i_c = ii.view(B, Kf, 1, Ki).expand(B, Kf, Kt, Ki)
            lpt_c = tv.view(B, Kf, Kt, 1).expand(B, Kf, Kt, Ki)
            lpi_c = iv.view(B, Kf, 1, Ki).expand(B, Kf, Kt, Ki)
            M = Kf * Kt * Ki
            comp = comp.reshape(B, M); f_c = f_c.reshape(B, M); t_c = t_c.reshape(B, M)
            i_c = i_c.reshape(B, M); lpf_c = lpf.reshape(B, M)
            lpt_c = lpt_c.reshape(B, M); lpi_c = lpi_c.reshape(B, M)
            if self.mfd_catalog_constraint:
                comp = comp.masked_fill(~self._valid_tuple_mask(f_c, t_c, i_c), float("-inf"))
            kk = min(Kp, M)
            sc_pool, pidx = comp.topk(kk, dim=-1)
            g = lambda x: torch.gather(x, 1, pidx)
            pf[:, b, :kk] = g(f_c); pt[:, b, :kk] = g(t_c); pi[:, b, :kk] = g(i_c)
            psc[:, b, :kk] = sc_pool
            plpf[:, b, :kk] = g(lpf_c); plpt[:, b, :kk] = g(lpt_c); plpi[:, b, :kk] = g(lpi_c)

        i16 = lambda x: x.to(torch.int16).cpu()
        i32 = lambda x: x.to(torch.int32).cpu()
        f16 = lambda x: x.to(torch.float16).cpu()
        if user_id is None:
            uid = torch.zeros(B, dtype=torch.int32)
        else:
            uid = i32(user_id[:, 0] if user_id.dim() > 1 else user_id)
        rec = {
            "user_id": uid,
            "gt_b": i16(gt_b), "gt_f": i16(gt_f), "gt_t": i16(gt_t), "gt_i": i16(gt_i),
            "beh_logp": f16(beh_logp),
            "gt_in_catalog": gt_in_catalog.to(torch.int8).cpu(),
            "f_star_rank": i32(f_star_rank), "f_star_logp": f16(f_star_logp),
            "tf_t_rank": i32(tf_t_rank), "tf_i_rank": i32(tf_i_rank),
            "tf_t_logp": f16(tf_t_logp), "tf_i_logp": f16(tf_i_logp),
            "pool_f": i16(pf), "pool_t": i16(pt), "pool_i": i16(pi),
            "pool_sidscore": f16(psc), "pool_lpf": f16(plpf),
            "pool_lpt": f16(plpt), "pool_lpi": f16(plpi),
        }
        # Stage-1 reranker dump: attach the frozen encoder memory (B,S,d) + mask (B,S). fp16
        # keeps it compact (~40KB/event at S=160,d=128); the reranker cross-attends into it.
        # Also store the raw history token sequence (input_ids: interleaved [b,f,t,i,...], pad=-1)
        # so the reranker can build per-candidate ENGAGEMENT features (hard tuple-match + soft
        # cosine-match against the user's history, per behavior). ~S int16 = negligible size.
        if self.mfd_dump_memory:
            rec["enc_mem"] = f16(enc)
            rec["enc_mask"] = emask.to(torch.int8).cpu()
            rec["hist_ids"] = input_ids.to(torch.int16).cpu()
        return rec

    def eval_step(self, batch, loss_to_aggregate):
        if not self.mfd_audit_dump:
            return super().eval_step(batch, loss_to_aggregate)
        if self._audit_n >= self.mfd_audit_max_events:
            return  # dump full — skip the (heavy) generation entirely
        model_input, label_data = batch
        inputs = {
            self.feature_to_model_input_map.get(k, k): v
            for k, v in model_input.transformed_sequences.items()
        }
        labels = list(label_data.labels.values())[0].reshape(model_input.mask.size(0), -1)
        rec = self._audit_forward(
            mask=model_input.mask, input_ids=inputs["input_ids"],
            user_id=inputs.get("user_id"), labels=labels.to(model_input.mask.device),
        )
        # stratified keep: every non-pv event; pv only up to pv_cap; overall cap max_events.
        gt_b = rec["gt_b"].long()
        keep = torch.zeros(gt_b.shape[0], dtype=torch.bool)
        for idx in range(gt_b.shape[0]):
            if self._audit_n >= self.mfd_audit_max_events:
                break
            if gt_b[idx].item() == self.mfd_audit_pv_id:
                if self._audit_pv >= self.mfd_audit_pv_cap:
                    continue
                self._audit_pv += 1
            keep[idx] = True
            self._audit_n += 1
        if keep.any():
            for k, v in rec.items():
                self._audit_cols[k].append(v[keep])

    def on_test_epoch_end(self):
        # Non-audit runs MUST fall through to the base hook that logs test/loss + metrics.
        if not self.mfd_audit_dump:
            return super().on_test_epoch_end()
        if not self._audit_cols:
            return
        cols = {k: torch.cat(v, dim=0) for k, v in self._audit_cols.items()}
        meta = {
            "nb": self.num_behaviors, "pv_id": self.mfd_audit_pv_id,
            "k0": self._k0, "k1": self._k1, "k2": self._k2,
            "Kf": min(self.mfd_top_k_shared, self._k0),
            "Kt": min(self.mfd_top_k_text, self._k1),
            "Ki": min(self.mfd_top_k_image, self._k2),
            "Kpool": self.mfd_pool_size, "tau": self.mfd_tau,
            "w_b_run": self.mfd_behavior_weight, "buy_id": self.buy_behavior_id,
            "d_model": self.embedding_dim, "has_memory": bool(self.mfd_dump_memory),
            "wide_catalog": bool(self.mfd_wide_catalog_pool),
        }
        out = {"cols": cols, "meta": meta}
        torch.save(out, self.mfd_audit_path)
        n = cols["gt_b"].shape[0]
        n_invalid = int((cols["gt_in_catalog"] == 0).sum())
        print(f"[MFD audit] wrote {n:,} events -> {self.mfd_audit_path}", flush=True)
        print(f"[MFD audit] CATALOG UNIT TEST: GT tuples NOT in catalog = {n_invalid} / {n} "
              f"({100.0 * n_invalid / max(n, 1):.4f}%) -- expected 0 if the constraint hash is correct.",
              flush=True)
        if n_invalid > 0:
            print("[MFD audit] ** catalog hash is WRONG (a real GT tuple is missing). **", flush=True)
        else:
            print("[MFD audit] catalog OK -> any 'GT not in pool' is POOL TRUNCATION (rank>Kpool), "
                  "not a catalog bug.", flush=True)