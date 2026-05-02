"""
Verification script for Semantic IDs (SIDs) from VQ-based models (MD-RQVAE, TIGER, etc.)

Supports any number of SID levels (3-level MDRQVAE, 4-level TIGER, etc.)

Usage:
    python src/utils/verify_sids.py \
        --pkl_path /scratch/yw8866/logs/inference/runs/2026-04-26/mdrqvae_1e_256/pickle/merged_predictions.pkl \
        --codebook_width 256

    # For TIGER-style SIDs with different key names:
    python src/utils/verify_sids.py \
        --pkl_path /path/to/predictions.pkl \
        --codebook_width 256 \
        --key_name user_id \
        --sid_name semantic_id
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def load_sids(pkl_path: str, key_name: str, sid_name: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    item_ids, sids = [], []
    for row in data:
        item_ids.append(row[key_name])
        sid = row[sid_name]
        if isinstance(sid, torch.Tensor):
            sids.append(sid.cpu().tolist())
        elif hasattr(sid, "tolist"):
            sids.append(sid.tolist())
        else:
            sids.append(list(sid))

    return item_ids, sids


def _header(title):
    print(f"\n{'='*60}")
    print(title)
    print("="*60)


# ---------------------------------------------------------------------------
# Test 1: Active ratio (index collapse detection)
# ---------------------------------------------------------------------------
def test_active_ratio(sids: list, codebook_width: int) -> dict:
    """
    For each codebook level, how many of the codebook_width codes are ever assigned?
    Active ratio < 50% is a strong signal of index collapse.
    """
    _header("1. ACTIVE RATIO TEST  (index collapse detection)")
    arr = np.array(sids, dtype=np.int64)
    num_levels = arr.shape[1]
    results = {}
    for lvl in range(num_levels):
        n_active = len(np.unique(arr[:, lvl]))
        ratio = n_active / codebook_width
        flag = "✓" if ratio >= 0.5 else "⚠ POSSIBLE COLLAPSE"
        print(f"  Level {lvl}: {n_active:>4d}/{codebook_width} codes active  ({ratio:.1%})  {flag}")
        results[lvl] = {"active_count": n_active, "active_ratio": ratio}
    return results


# ---------------------------------------------------------------------------
# Test 2: Collision rate
# ---------------------------------------------------------------------------
def test_collision_rate(item_ids: list, sids: list) -> dict:
    """
    Fraction of items that share a SID tuple with at least one other item.
    Perfect model: 0% collision (each item gets a unique code sequence).
    """
    _header("2. COLLISION RATE TEST")
    N = len(item_ids)
    tuples = [tuple(s) for s in sids]
    counter = Counter(tuples)

    unique_sids = len(counter)
    colliding_sids = {k: v for k, v in counter.items() if v > 1}
    items_in_collision = sum(colliding_sids.values())
    collision_rate = items_in_collision / N

    print(f"  Total items:            {N:>10,}")
    print(f"  Unique SID tuples:      {unique_sids:>10,}")
    print(f"  Collision rate:         {collision_rate:>10.2%}")
    print(f"  SID tuples w/ collisions: {len(colliding_sids):>8,}")
    print(f"  Items in any collision: {items_in_collision:>10,}  ({items_in_collision / N:.2%})")

    if colliding_sids:
        sizes = sorted(colliding_sids.values(), reverse=True)
        print(f"  Largest collision group: {sizes[0]} items share the same SID")
        print(f"  Top-5 collision sizes:   {sizes[:5]}")
        # Histogram of collision sizes
        size_dist = Counter(sizes)
        print("  Collision size distribution (size: count):")
        for sz in sorted(size_dist)[:10]:
            print(f"    {sz} items sharing same SID: {size_dist[sz]:,} groups")

    return {
        "unique_sids": unique_sids,
        "collision_rate": collision_rate,
        "num_colliding_sids": len(colliding_sids),
        "items_in_collision": items_in_collision,
    }


# ---------------------------------------------------------------------------
# Test 3: Per-level codebook entropy
# ---------------------------------------------------------------------------
def test_codebook_entropy(sids: list, codebook_width: int) -> dict:
    """
    Shannon entropy of the code distribution at each level.
    Ideal (uniform): log2(codebook_width) bits.
    Low entropy → a few codes dominate, most are wasted.
    """
    _header("3. CODEBOOK ENTROPY  (distribution uniformity)")
    arr = np.array(sids, dtype=np.int64)
    num_levels = arr.shape[1]
    max_entropy = np.log2(codebook_width)
    print(f"  Max theoretical entropy: {max_entropy:.3f} bits  (uniform distribution)")
    results = {}
    for lvl in range(num_levels):
        counts = np.bincount(arr[:, lvl], minlength=codebook_width).astype(float)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        entropy = float(-np.sum(probs_nz * np.log2(probs_nz)))
        efficiency = entropy / max_entropy
        flag = "✓" if efficiency >= 0.7 else "⚠ LOW"
        print(f"  Level {lvl}: entropy={entropy:.3f} bits  efficiency={efficiency:.1%}  {flag}")
        results[lvl] = {"entropy": entropy, "efficiency": efficiency}
    return results


# ---------------------------------------------------------------------------
# Test 4: Theoretical vs actual SID coverage
# ---------------------------------------------------------------------------
def test_coverage(sids: list, codebook_width: int) -> dict:
    """
    Out of codebook_width^num_levels possible SID tuples, how many are used?
    Also reports the ratio unique_SIDs / num_items (effective uniqueness).
    """
    _header("4. SID COVERAGE")
    arr = np.array(sids, dtype=np.int64)
    N, num_levels = arr.shape
    theoretical_max = codebook_width ** num_levels
    unique_sids = len(set(map(tuple, sids)))

    print(f"  Num levels:               {num_levels}")
    print(f"  Codebook width:           {codebook_width}")
    print(f"  Theoretical capacity:     {theoretical_max:>20,}")
    print(f"  Items:                    {N:>20,}")
    print(f"  Unique SIDs used:         {unique_sids:>20,}")
    print(f"  Uniqueness (unique/items): {unique_sids / N:.4f}")
    print(f"  Capacity utilisation:      {unique_sids / theoretical_max:.2e}")
    return {
        "theoretical_max": theoretical_max,
        "unique_sids": unique_sids,
        "uniqueness_ratio": unique_sids / N,
    }


# ---------------------------------------------------------------------------
# Test 5: Inter-level correlation
# ---------------------------------------------------------------------------
def test_level_independence(sids: list) -> dict:
    """
    If two levels frequently agree on the same code index, they are correlated
    (redundant). Independent levels should agree at roughly 1/codebook_width rate.
    """
    arr = np.array(sids, dtype=np.int64)
    num_levels = arr.shape[1]
    if num_levels < 2:
        return {}
    _header("5. LEVEL INDEPENDENCE  (inter-level correlation)")
    results = {}
    for i in range(num_levels):
        for j in range(i + 1, num_levels):
            same_rate = float(np.mean(arr[:, i] == arr[:, j]))
            flag = "✓ independent" if same_rate < 0.1 else "⚠ correlated"
            print(f"  Level {i} vs Level {j}: identical code rate = {same_rate:.2%}  {flag}")
            results[(i, j)] = same_rate
    return results


# ---------------------------------------------------------------------------
# Test 6: Item integrity (duplicate item_ids)
# ---------------------------------------------------------------------------
def test_item_integrity(item_ids: list, sids: list) -> dict:
    """
    Checks that each item_id maps to exactly one SID tuple.
    If the same item appears multiple times with different SIDs, something went
    wrong in data loading or deduplication.
    """
    _header("6. ITEM INTEGRITY CHECK  (duplicate item IDs)")
    N = len(item_ids)
    item_to_sids: dict = defaultdict(set)
    for iid, sid in zip(item_ids, sids):
        item_to_sids[iid].add(tuple(sid))

    unique_items = len(item_to_sids)
    inconsistent = {k: v for k, v in item_to_sids.items() if len(v) > 1}

    print(f"  Total entries:            {N:>10,}")
    print(f"  Unique item IDs:          {unique_items:>10,}")
    dup_flag = "✓ none" if not inconsistent else f"⚠ {len(inconsistent):,} items have conflicting SIDs"
    print(f"  Items with multiple SIDs: {dup_flag}")
    if inconsistent:
        example_key = next(iter(inconsistent))
        print(f"  Example conflict: item_id={example_key} → {inconsistent[example_key]}")

    return {"unique_items": unique_items, "inconsistent_items": len(inconsistent)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Verify Semantic IDs from VQ-based models (MD-RQVAE, TIGER, ...)."
    )
    parser.add_argument(
        "--pkl_path", type=str, required=True,
        help="Path to merged_predictions.pkl"
    )
    parser.add_argument(
        "--codebook_width", type=int, default=256,
        help="Number of codes per codebook level (default: 256)"
    )
    parser.add_argument(
        "--key_name", type=str, default="item_id",
        help="Dict key for the item ID field (default: item_id)"
    )
    parser.add_argument(
        "--sid_name", type=str, default="cluster_ids",
        help="Dict key for the SID field (default: cluster_ids)"
    )
    args = parser.parse_args()

    pkl_path = Path(args.pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    print(f"\nLoading SIDs from: {pkl_path}")
    item_ids, sids = load_sids(str(pkl_path), args.key_name, args.sid_name)
    num_levels = len(sids[0]) if sids else 0
    print(f"Loaded {len(item_ids):,} items  |  {num_levels} SID levels per item  |  codebook_width={args.codebook_width}")

    test_active_ratio(sids, args.codebook_width)
    test_collision_rate(item_ids, sids)
    test_codebook_entropy(sids, args.codebook_width)
    test_coverage(sids, args.codebook_width)
    test_level_independence(sids)
    test_item_integrity(item_ids, sids)

    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()