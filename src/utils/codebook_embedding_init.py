"""
Utility for injecting MD-RQ-VAE codebook geometry into the TIGER-MB embedding table.

Global vocab layout built here:
    [0 : K_f)               shared SIDs   (C_f, 2048 entries)
    [K_f : K_f+K_t)         text SIDs     (C_t, 1024 entries)
    [K_f+K_t : K_f+K_t+K_i) image SIDs   (C_i, 1024 entries)
    [K_f+K_t+K_i : V)       behavior tokens (num_behaviors entries, random init)

Up-projection W_up ∈ R^{d_vae × d_model} uses orthogonal initialisation so that
column norms are preserved and pairwise cosine similarities in the projected space
equal those in the original d_vae space (up to the truncation inherent in projecting
from high to low dimension).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Checkpoint key names for the three codebooks inside an MDRQVAE checkpoint
# ---------------------------------------------------------------------------
_CENTROID_KEYS = {
    "shared": "shared_codebook.centroids",
    "text":   "text_codebook.centroids",
    "image":  "image_codebook.centroids",
}


def load_codebook_centroids(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the three frozen codebook centroid matrices from a Lightning checkpoint.

    Returns:
        c_f: (K_f, d_vae)  shared codebook
        c_t: (K_t, d_vae)  text-specific codebook
        c_i: (K_i, d_vae)  image-specific codebook
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    c_f = sd[_CENTROID_KEYS["shared"]].float()
    c_t = sd[_CENTROID_KEYS["text"]].float()
    c_i = sd[_CENTROID_KEYS["image"]].float()

    return c_f, c_t, c_i


def build_codebook_init_embedding(
    checkpoint_path: str,
    d_model: int,
    num_behaviors: int = 4,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[int]]:
    """
    Build the unified embedding table by orthogonally up-projecting codebook vectors.

    Args:
        checkpoint_path: Path to the MD-RQ-VAE Lightning checkpoint (.ckpt).
        d_model:         Target embedding dimension (LLM hidden size).
        num_behaviors:   Number of special behavior tokens appended at the end.
        device:          Device for intermediate computation ('cpu' recommended at init).

    Returns:
        E_unified: (K_f + K_t + K_i + num_behaviors, d_model)  — initial embedding weights.
        codebook_sizes: [K_f, K_t, K_i]  — sizes actually found in the checkpoint.
    """
    c_f, c_t, c_i = load_codebook_centroids(checkpoint_path, device=device)
    d_vae = c_f.shape[1]
    codebook_sizes = [c_f.shape[0], c_t.shape[0], c_i.shape[0]]
    total_sid = sum(codebook_sizes)
    vocab_size = total_sid + num_behaviors

    # --- Orthogonal up-projection W_up: (d_vae, d_model) ---
    # torch.nn.init.orthogonal_ produces column-orthonormal bases, which is an
    # isometry from the column space of W_up back into R^d_vae.  Cosine similarities
    # between projected vectors equal those in the original d_vae space.
    w_up = torch.empty(d_vae, d_model, device=device)
    torch.nn.init.orthogonal_(w_up)

    # --- Assemble E_unified block by block ---
    e_unified = torch.empty(vocab_size, d_model, device=device)
    with torch.no_grad():
        offset = 0
        for codebook in (c_f, c_t, c_i):
            k = codebook.shape[0]
            e_unified[offset : offset + k] = codebook.to(device) @ w_up
            offset += k

        # Behavior token block: standard random init (same as nn.Embedding default)
        torch.nn.init.normal_(e_unified[offset:])

    return e_unified, codebook_sizes