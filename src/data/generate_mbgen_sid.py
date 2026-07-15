"""Generate MBGen (MMMB-Genrec)-style semantic IDs for the GRID/Tmall catalogue.

This is a faithful re-implementation (not a copy) of the QAE_Kmeans tokenizer from
/scratch/yw8866/MMMB-Genrec (tokenizer/tokenizer.py, tokenizer/models/{QAE,layers}.py,
tokenizer/trainer.py), adapted to run at Tmall scale (~5.4M items) and to emit the SID
tensor in the exact format GRID's tiger_mb pipeline consumes.

Algorithm (2 tokens per item):
  Level 1 (QAE): MLP encoder -> single EMA-updated VQ codebook (kmeans-initialised,
                 latent_dim) -> MLP decoder, trained with MSE reconstruction + beta*commit.
                 The nearest codebook entry is level-1 code id1; residual = encoded - quant.
  Level 2 (conditional KMeans): group items by id1, run KMeans(num_id2) on the residuals
                 WITHIN each level-1 group (local codes). expand_id2=False keeps id2 in
                 [0, num_id2); True makes them globally unique.

Differences from MBGen upstream (all forced by scale; documented in AGENTS.md):
  * input embedding is the concatenation of GRID's SigLIP text + image vectors (2304-d),
    standardised, instead of MBGen's single opaque embedding.pkl.
  * codebook kmeans-init runs on a subsample (MiniBatchKMeans) instead of the full set.
  * level-2 uses MiniBatchKMeans for groups larger than num_id2, and a degenerate
    arange assignment (one cluster per item) for groups with <= num_id2 items so that
    sklearn never sees n_clusters > n_samples.

Output: torch.int64 tensor of shape [2, num_items] saved to <output_dir>/semantic_ids.pt,
column index == item_id, gaps / items missing a modality left as [0, 0] (padding), exactly
matching the layout of the MD-RQ-VAE semantic_ids.pt (see verify_md_sid_2048_1024.out).

Usage:
  python src/data/generate_mbgen_sid.py --config configs/experiment/mbgen_sid_gen.yaml \
      [key=value ...]      # dotlist overrides, e.g. qae.epochs=60
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# QAE (MLP autoencoder + single EMA-VQ codebook)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, sizes, dropout=0.0):
        super().__init__()
        mods = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
            mods.append(nn.Linear(a, b))
            if i != len(sizes) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class EMAVectorQuantizer(nn.Module):
    """Single-codebook VQ with EMA updates and straight-through gradient.

    Mirrors MBGen's QuantizationLayer: embed is [dim, n_embed]; nearest code by squared
    L2; EMA on cluster_size / embed_avg; commitment loss = MSE(quantize.detach(), x).
    """

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.register_parameter("embed", nn.Parameter(torch.zeros(dim, n_embed)))
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", torch.zeros(dim, n_embed))

    def _dist(self, x):
        # x [B, dim] -> [B, n_embed]
        return (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

    def forward(self, x):
        with torch.no_grad():
            ind = (-self._dist(x.detach())).max(1).indices
        quantize = F.embedding(ind, self.embed.t())
        if self.training:
            # EMA codebook update. MUST be under no_grad on DETACHED activations: an
            # in-place add_ of a grad-requiring tensor onto a buffer gives the buffer a
            # grad_fn, which chains the autograd graph across iterations and never frees
            # the encoder activations (that is a multi-GB/step leak -> CUDA OOM).
            with torch.no_grad():
                xd = x.detach()
                onehot = F.one_hot(ind, self.n_embed).type(xd.dtype)
                self.cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay
                )
                self.embed_avg.mul_(self.decay).add_(
                    xd.t() @ onehot, alpha=1 - self.decay
                )
                n = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
                self.embed.data.copy_(self.embed_avg / cs.unsqueeze(0))
        commit_loss = F.mse_loss(quantize.detach(), x)
        quantize_ste = x + (quantize - x).detach()
        return quantize_ste, ind, commit_loss

    @torch.no_grad()
    def assign(self, x):
        ind = (-self._dist(x)).max(1).indices
        quantize = F.embedding(ind, self.embed.t())
        return ind, quantize

    @torch.no_grad()
    def init_from_kmeans(self, encoded_sample, device):
        km = MiniBatchKMeans(
            n_clusters=self.n_embed, n_init=3, max_iter=100, batch_size=10000
        ).fit(encoded_sample)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float, device=device)
        self.embed.data = centers.t().contiguous()
        self.embed_avg.data = centers.t().contiguous()
        counts = np.bincount(km.labels_, minlength=self.n_embed)
        self.cluster_size.data = torch.tensor(counts, dtype=torch.float, device=device)


class QAE(nn.Module):
    def __init__(self, input_dim, hidden, latent_dim, n_embed, dropout=0.0):
        super().__init__()
        self.encoder = MLP([input_dim] + list(hidden) + [latent_dim], dropout)
        self.vq = EMAVectorQuantizer(latent_dim, n_embed)
        self.decoder = MLP([latent_dim] + list(reversed(hidden)) + [input_dim], dropout)

    def forward(self, x):
        enc = self.encoder(x)
        quant_ste, ind, commit = self.vq(enc)
        return self.decoder(quant_ste), commit, ind


# ---------------------------------------------------------------------------
# Embedding assembly: concat SigLIP text + image aligned by item_id
# ---------------------------------------------------------------------------
def load_id_to_row(id_map_path):
    with open(id_map_path, "rb") as f:
        m = pickle.load(f)
    if isinstance(m, dict):
        return {int(k): int(v) for k, v in m.items()}
    return {int(item_id): row for row, item_id in enumerate(m)}


def open_embedding_array(npy_path, dim):
    """Open a SigLIP embedding file as a read-only (N, dim) array.

    Mirrors src.utils.file_utils.load_indexed_npy_embeddings: standard .npy files start
    with the numpy magic bytes; GRID's exports are raw float32 dumps (exact data size, no
    header) and must be opened as a plain memmap of shape (filesize/4/dim, dim).
    """
    with open(npy_path, "rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        return np.load(npy_path, mmap_mode="r", allow_pickle=False)
    n_rows = os.path.getsize(npy_path) // 4 // dim
    return np.memmap(npy_path, dtype=np.float32, mode="r", shape=(n_rows, dim))


def gather_rows(npy_path, rows, dim, chunk=200_000):
    """Fancy-index `rows` out of a (possibly mmap'd) array of shape (N, dim)."""
    arr = open_embedding_array(npy_path, dim)
    assert arr.shape[1] == dim, f"{npy_path} dim {arr.shape[1]} != {dim}"
    out = np.empty((len(rows), dim), dtype=np.float32)
    order = np.argsort(rows)  # sequential-ish reads improve mmap locality
    rows_sorted = np.asarray(rows)[order]
    for s in range(0, len(rows), chunk):
        e = min(s + chunk, len(rows))
        block = np.asarray(arr[rows_sorted[s:e]], dtype=np.float32)
        out[order[s:e]] = block
        if (s // chunk) % 10 == 0:
            log(f"    gathered {e:,}/{len(rows):,} from {os.path.basename(npy_path)}")
    return out


def build_embeddings(cfg):
    log("loading id maps ...")
    text_map = load_id_to_row(cfg.text_id_map_path)
    image_map = load_id_to_row(cfg.image_id_map_path)
    items = sorted(set(text_map) & set(image_map))
    log(
        f"items: text={len(text_map):,} image={len(image_map):,} "
        f"intersection={len(items):,}"
    )
    text_rows = [text_map[i] for i in items]
    image_rows = [image_map[i] for i in items]

    log("gathering text embeddings ...")
    Xt = gather_rows(cfg.text_embedding_path, text_rows, cfg.modality_dim)
    log("gathering image embeddings ...")
    Xi = gather_rows(cfg.image_embedding_path, image_rows, cfg.modality_dim)
    X = np.concatenate([Xt, Xi], axis=1)  # [M, 2*modality_dim]
    del Xt, Xi
    log(f"assembled embedding matrix {X.shape} ({X.nbytes / 1e9:.1f} GB)")

    # StandardScaler equivalent, in place to avoid a full copy.
    log("standardising (per-feature mean/std) ...")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X -= mean
    X /= std
    return np.asarray(items, dtype=np.int64), X


# ---------------------------------------------------------------------------
# QAE training + encoding
# ---------------------------------------------------------------------------
def train_qae(cfg, X, device):
    n, input_dim = X.shape
    model = QAE(
        input_dim, cfg.qae.hidden, cfg.qae.latent_dim, cfg.num_id1, cfg.qae.dropout
    ).to(device)
    log(f"QAE: {model}")

    # Optionally fit the QAE on a subsample. The autoencoder + codebook converge fine on a
    # few million items and this is the only GPU-hungry stage, so subsampling is what makes
    # a CPU-only run tractable. ALL items are still encoded afterwards (encode_all).
    train_sample = cfg.qae.get("train_sample", None)
    if train_sample and train_sample < n:
        sub = np.sort(
            np.random.default_rng(cfg.seed + 1).choice(n, int(train_sample), replace=False)
        )
        Xtrain = X[sub]
        log(f"training QAE on a {len(sub):,}-item subsample of {n:,}")
    else:
        Xtrain = X
        log(f"training QAE on all {n:,} items")

    # Codebook kmeans-init on a subsample of the *encoded* features.
    sample_n = min(cfg.qae.init_sample, n)
    idx = np.sort(np.random.default_rng(cfg.seed).choice(n, sample_n, replace=False))
    chunks = []
    with torch.no_grad():
        for s in range(0, sample_n, cfg.qae.encode_batch_size):
            e = min(s + cfg.qae.encode_batch_size, sample_n)
            xb = torch.from_numpy(X[idx[s:e]]).to(device)
            chunks.append(model.encoder(xb).cpu().numpy())
    enc_sample = np.concatenate(chunks, axis=0)
    del chunks
    log(f"kmeans-initialising codebook ({cfg.num_id1} codes on {sample_n:,} samples) ...")
    model.vq.init_from_kmeans(enc_sample, device)

    ds = TensorDataset(torch.from_numpy(Xtrain))
    dl = DataLoader(
        ds,
        batch_size=cfg.qae.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),  # pinning is a pure waste on CPU
    )
    opt = torch.optim.Adagrad(model.parameters(), lr=cfg.qae.lr)
    beta = cfg.qae.beta
    for epoch in range(cfg.qae.epochs):
        model.train()
        tot = rec = com = 0.0
        # True codebook usage: which codes were actually selected this epoch. (The EMA
        # cluster_size settles at a sum equal to the batch size, so thresholding it is a
        # batch-size artefact, not a usage measure.)
        used = torch.zeros(cfg.num_id1, dtype=torch.bool, device=device)
        for (xb,) in dl:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            recon, commit, ind = model(xb)
            rloss = F.mse_loss(recon, xb)
            loss = rloss + beta * commit
            loss.backward()
            opt.step()
            tot += loss.item()
            rec += rloss.item()
            com += commit.item()
            used[ind] = True
        nb = len(dl)
        active = int(used.sum().item())
        log(
            f"epoch {epoch + 1}/{cfg.qae.epochs} loss={tot / nb:.4f} "
            f"rec={rec / nb:.4f} commit={com / nb:.4f} active_codes={active}/{cfg.num_id1}"
        )
    return model


@torch.no_grad()
def encode_all(model, X, device, batch_size):
    model.eval()
    n = X.shape[0]
    id1 = np.empty(n, dtype=np.int64)
    residual = np.empty((n, model.vq.dim), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.from_numpy(X[s:e]).to(device)
        enc = model.encoder(xb)
        ind, quant = model.vq.assign(enc)
        id1[s:e] = ind.cpu().numpy()
        residual[s:e] = (enc - quant).cpu().numpy()
    return id1, residual


# ---------------------------------------------------------------------------
# Level-2: conditional KMeans on residuals within each level-1 group
# ---------------------------------------------------------------------------
def _cluster_group(args):
    gid, res, num_id2, seed = args
    n = res.shape[0]
    if n <= num_id2:
        # Degenerate: each item gets its own local code (unique, no collision).
        return gid, np.arange(n, dtype=np.int64)
    # Pin each worker to a single BLAS/OpenMP thread. Without this, N worker processes each
    # spawn N OpenMP threads and thrash the cores (48 x 48 = 2304 threads on 48 cores).
    with threadpool_limits(limits=1):
        km = MiniBatchKMeans(
            n_clusters=num_id2,
            n_init=1,  # n_init=3 triples cost for negligible gain at this k
            max_iter=50,
            batch_size=min(4096, n),
            init_size=min(n, 3 * num_id2),
            max_no_improvement=20,
            reassignment_ratio=0.0,  # no low-count centre churn
            random_state=seed,
        ).fit(res)
    return gid, km.labels_.astype(np.int64)


def conditional_kmeans(cfg, id1, residual):
    order = np.argsort(id1, kind="stable")
    id1_sorted = id1[order]
    boundaries = np.searchsorted(id1_sorted, np.arange(cfg.num_id1 + 1))
    tasks = []
    for g in range(cfg.num_id1):
        lo, hi = boundaries[g], boundaries[g + 1]
        if hi > lo:
            tasks.append((g, residual[order[lo:hi]], cfg.num_id2, cfg.seed))
    log(f"conditional kmeans over {len(tasks)} non-empty groups ...")

    id2 = np.zeros(len(id1), dtype=np.int64)
    group_slices = {g: (boundaries[g], boundaries[g + 1]) for g in range(cfg.num_id1)}
    done = 0
    t0 = time.time()

    # MUST use 'spawn', not the default 'fork'. By this point the parent has run CPU torch
    # training, so an OpenMP runtime is live in-process -- and libgomp is NOT fork-safe: a
    # forked child hangs forever the first time it enters a parallel region, which is
    # exactly what sklearn's KMeans does. Forking here deadlocks every worker on group 1.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_cluster_group, t) for t in tasks]
        for fut in as_completed(futs):
            g, labels = fut.result()
            lo, hi = group_slices[g]
            local = labels.copy()
            if cfg.expand_id2:
                local = local + g * cfg.num_id2
            id2[order[lo:hi]] = local
            done += 1
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                log(
                    f"    {done}/{len(tasks)} groups clustered "
                    f"({el / 60:.1f} min elapsed, ~{eta / 60:.1f} min left)"
                )
    return id2


# ---------------------------------------------------------------------------
# Level-3: dedup counter (MBGen's uniqueness mechanism)
# ---------------------------------------------------------------------------
def dedup_token(id1, id2):
    """Third SID token: a running counter within each (id1, id2) bucket.

    Faithful to MBGen's QAE_Kmeans_item_Tokenizer: items that share a (id1, id2) pair are
    numbered 0, 1, 2, ... so every item gets a distinct 3-token SID (a TIGER-style dedup
    token). Returns id3 and the dedup vocab size (= largest bucket == max(id3)+1).
    """
    key = id1.astype(np.int64) * (int(id2.max()) + 1) + id2.astype(np.int64)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    change = np.ones(len(sorted_key), dtype=bool)
    change[1:] = sorted_key[1:] != sorted_key[:-1]
    grp_start = np.flatnonzero(change)
    starts_per_pos = np.repeat(
        grp_start, np.diff(np.append(grp_start, len(sorted_key)))
    )
    within = np.arange(len(sorted_key)) - starts_per_pos  # 0-based rank in bucket
    id3 = np.empty(len(id1), dtype=np.int64)
    id3[order] = within
    return id3, int(id3.max()) + 1


# ---------------------------------------------------------------------------
# Verification (mirrors verify_md_sid_2048_1024.out)
# ---------------------------------------------------------------------------
def verify(sid, items):
    levels = sid.size(0)
    codes = [sid[h, items] for h in range(levels)]
    n = len(items)
    log("=" * 60)
    log("VERIFICATION")
    log(f"  items: {n:,}   SID levels: {levels}")
    for h, c in enumerate(codes):
        log(f"  level {h} active codes: {torch.unique(c).numel()}   (max id {int(c.max())})")
    # Combined-key collision over all levels.
    key = torch.zeros(n, dtype=torch.int64)
    for c in codes:
        key = key * (int(c.max()) + 1) + c.to(torch.int64)
    uniq = torch.unique(key).numel()
    log(f"  unique SID tuples: {uniq:,}   collision rate: {1 - uniq / n:.4f}")
    log(f"  uniqueness (unique/items): {uniq / n:.4f}")
    log("=" * 60)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotlist overrides key=value")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    log("config:\n" + OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # device: auto | cuda | cpu. The GPU only accelerates QAE training; the embedding
    # gather and the level-2 conditional KMeans are CPU/IO-bound either way. A CPU-only
    # run is fully supported -- pair it with qae.train_sample to keep training tractable.
    want = str(cfg.get("device", "auto"))
    if want == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = want
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but no CUDA device is visible")
    if device == "cpu":
        torch.set_num_threads(int(cfg.num_workers))
    log(f"device: {device} (torch threads={torch.get_num_threads()})")

    # Stage-1 cache: (items, id1, residual, codebook). Everything before the level-2 KMeans
    # -- the ~50GB embedding gather, QAE training, and encoding -- is deterministic given
    # the config, so cache it. Lets you re-run/tune ONLY the conditional-KMeans stage
    # without repeating an hour of I/O. Delete the file (or set use_encode_cache=false)
    # to force a full rebuild.
    os.makedirs(cfg.output_dir, exist_ok=True)
    cache_path = os.path.join(cfg.output_dir, "encode_cache.pt")
    use_cache = bool(cfg.get("use_encode_cache", True))

    if use_cache and os.path.exists(cache_path):
        log(f"loading stage-1 cache: {cache_path} (skipping gather / QAE / encode)")
        # weights_only=False: the cache holds numpy arrays (id1/residual), which PyTorch
        # 2.6's default weights_only=True refuses to unpickle. This is our own file, trusted.
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        items, id1, residual = cache["items"], cache["id1"], cache["residual"]
        codebook = cache["codebook"]
        log(f"cached: {len(items):,} items, residual {residual.shape}")
    else:
        items, X = build_embeddings(cfg)
        model = train_qae(cfg, X, device)
        log("encoding all items (level-1 code + residual) ...")
        id1, residual = encode_all(model, X, device, cfg.qae.encode_batch_size)
        del X
        codebook = model.vq.embed.detach().cpu()
        torch.save(
            {"items": items, "id1": id1, "residual": residual, "codebook": codebook},
            cache_path,
        )
        log(f"saved stage-1 cache: {cache_path}")

    id2 = conditional_kmeans(cfg, id1, residual)

    # Level 3: MBGen's dedup counter -> guarantees a unique SID per item. Optional so a
    # pure 2-token variant is still reachable (add_dedup_token=false), but ON by default
    # because it is how MBGen actually assigns IDs (see QAE_Kmeans_item_Tokenizer).
    add_dedup = bool(cfg.get("add_dedup_token", True))
    if add_dedup:
        id3, dedup_vocab = dedup_token(id1, id2)
        log(
            f"dedup token: max bucket = {dedup_vocab} items sharing one (id1,id2) "
            f"-> level-2 vocab must be >= {dedup_vocab}"
        )
        levels = [id1, id2, id3]
    else:
        levels = [id1, id2]

    num_items = max(int(cfg.num_items), int(items.max()) + 1)
    log(f"writing SID tensor [{len(levels)}, {num_items:,}] (column == item_id) ...")
    sid = torch.zeros(len(levels), num_items, dtype=torch.int64)
    items_t = torch.from_numpy(items)
    for h, lv in enumerate(levels):
        sid[h, items_t] = torch.from_numpy(lv)

    verify(sid, items_t)

    # Print the exact tiger_mb overrides so there is no guessing about codebook_sizes.
    cb = [int(cfg.num_id1), int(sid[1, items_t].max()) + 1]
    if add_dedup:
        cb.append(int(sid[2, items_t].max()) + 1)
    log(
        f"=> train with: num_hierarchies={len(levels)} "
        f"codebook_sizes={cb}"
    )

    out_path = os.path.join(cfg.output_dir, "semantic_ids.pt")
    torch.save(sid, out_path)
    log(f"saved: {out_path}  shape={tuple(sid.shape)} dtype={sid.dtype}")
    # Also persist the raw QAE codebook + id assignments for reproducibility / analysis.
    artifacts = {"item_ids": items, "id1": id1, "id2": id2, "codebook": codebook}
    if add_dedup:
        artifacts["id3"] = id3
    torch.save(artifacts, os.path.join(cfg.output_dir, "mbgen_sid_artifacts.pt"))
    log("done.")


if __name__ == "__main__":
    sys.exit(main())