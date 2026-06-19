"""
Convert Tmall interaction logs into GZIP TFRecord files for multi-behavior TIGER training.

Each TFRecord row = one user's full chronological interaction sequence:
  - sequence_data : int64 list  (item IDs, sorted by timestamp)
  - behavior_data : int64 list  (behavior id per item)
  - user_id       : int64 scalar

Raw Tmall behavior strings -> behavior id (semantics kept consistent downstream):
    click   -> 0  (pv,   page view)
    collect -> 1  (fav,  favorite)
    cart    -> 2  (cart, add-to-cart)
    alipay  -> 3  (buy,  purchase)

Users are split deterministically (by user_id % 100) into training / evaluation / testing.

Optional class-imbalance rebalancing (TRAINING split only; eval/test stay natural):
    --augment_mode user_drop         drop the most pv-dominated whole users until the
                                     training pv share hits --target_pv_ratio.
    --augment_mode event_downsample  thin pv/click events within each training sequence
                                     (keeping 100% of cart/collect/alipay) until the
                                     training pv share hits --target_pv_ratio.

Usage:
    python src/data/build_tmall_tfrecords_mb.py \
        --valid_data_dir /scratch/yw8866/rec-tmall/valid_data \
        --output_dir     /scratch/yw8866/rec-tmall/tfrecords_mb \
        --min_seq_len    5 \
        --users_per_file 5000 \
        --train_frac     0.8 \
        --val_frac       0.1 \
        [--augment_mode {user_drop,event_downsample} --target_pv_ratio 0.80]
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

SEP = "\x01"
LOG_FILES = [
    "tianchi_2014002_rec_tmall_log_parta.txt",
    "tianchi_2014002_rec_tmall_log_partb.txt",
    "tianchi_2014002_rec_tmall_log_partc.txt",
]

# Raw behavior strings as they appear in the Tmall logs (lowercase).
BEHAVIOR_MAP = {"click": 0, "collect": 1, "cart": 2, "alipay": 3}
# Human-readable id names used only for reporting.
BEHAVIOR_ID_NAMES = {0: "pv (click)", 1: "fav (collect)", 2: "cart", 3: "buy (alipay)"}
# Sentinel id for rows whose behavior string is not in BEHAVIOR_MAP.
# Kept OUT of the real classes so a future label mismatch is visible, not silently
# folded into pv. Rows with this id are reported and dropped before writing.
UNKNOWN_BEHAVIOR = -1


def parse_ts(ts_str: str) -> int:
    """'YYYY-MM-DD HH:MM:SS' → YYYYMMDDHHMMSS integer (lexicographic order preserved)."""
    return int(ts_str[:19].replace("-", "").replace(" ", "").replace(":", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_data_dir", required=True,
                        help="Directory containing the filtered log files")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory; training/ evaluation/ testing/ created inside")
    parser.add_argument("--min_seq_len", type=int, default=5,
                        help="Drop users with fewer than this many interactions (default: 5)")
    parser.add_argument("--max_seq_len", type=int, default=200,
                        help="Drop users with more than this many interactions (default: 200)")
    parser.add_argument("--users_per_file", type=int, default=5000,
                        help="Number of user sequences written per TFRecord file (default: 5000)")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of users for training split (default: 0.8)")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of users for evaluation split (default: 0.1)")
    parser.add_argument("--augment_mode", choices=["none", "user_drop", "event_downsample"],
                        default="none",
                        help="Rebalancing strategy applied to the TRAINING split only:\n"
                             "  none             : no rebalancing (natural distribution)\n"
                             "  user_drop        : drop the most pv-dominated whole users until\n"
                             "                     the training pv share hits --target_pv_ratio.\n"
                             "                     Sequences stay 100%% real; some buy/cart lost.\n"
                             "  event_downsample : thin pv/click events within each training\n"
                             "                     sequence (keep 100%% of cart/collect/alipay)\n"
                             "                     until pv share hits --target_pv_ratio.")
    parser.add_argument("--target_pv_ratio", type=float, default=0.80,
                        help="Target pv/click share of the training split when --augment_mode "
                             "is not 'none' (default: 0.80)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible rebalancing (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Behavior map: {BEHAVIOR_MAP}")
    if args.augment_mode == "none":
        print("Rebalancing: OFF (natural distribution)")
    else:
        print(f"Rebalancing: {args.augment_mode}  (training split only, "
              f"target_pv_ratio={args.target_pv_ratio}, seed={args.seed})")

    # ------------------------------------------------------------------
    # Phase 1: Read all log files into four compact numpy arrays
    # ------------------------------------------------------------------
    print("\nPhase 1: Reading interaction logs...", flush=True)
    MAX_ROWS = 1_200_000_000
    uid_arr  = np.empty(MAX_ROWS, dtype=np.int32)
    iid_arr  = np.empty(MAX_ROWS, dtype=np.int32)
    ts_arr   = np.empty(MAX_ROWS, dtype=np.int64)
    bhv_arr  = np.empty(MAX_ROWS, dtype=np.int8)
    n = 0
    skipped_parse = 0
    skipped_unknown_bhv = 0
    unknown_bhv_strings: dict = {}  # raw string -> count, for diagnostics

    for fname in LOG_FILES:
        fpath = os.path.join(args.valid_data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] not found: {fpath}")
            continue
        print(f"  Reading {fname} ...", flush=True)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split(SEP, 3)
                if len(parts) < 4:
                    skipped_parse += 1
                    continue
                try:
                    btype = parts[2].strip()
                    bhv_id = BEHAVIOR_MAP.get(btype, UNKNOWN_BEHAVIOR)
                    # Drop rows with an unrecognised behavior string instead of
                    # silently folding them into a real class. Track them so the
                    # final report shows whether any slipped through.
                    if bhv_id == UNKNOWN_BEHAVIOR:
                        skipped_unknown_bhv += 1
                        unknown_bhv_strings[btype] = unknown_bhv_strings.get(btype, 0) + 1
                        continue
                    item_id = int(parts[0].strip())
                    uid_str = parts[1].strip()
                    user_id = int(uid_str[1:] if uid_str.startswith("u") else uid_str)
                    ts      = parse_ts(parts[3])
                    uid_arr[n] = user_id
                    iid_arr[n] = item_id
                    ts_arr[n]  = ts
                    bhv_arr[n] = bhv_id
                    n += 1
                except (ValueError, IndexError):
                    skipped_parse += 1
        print(f"    Rows so far: {n:,}", flush=True)

    uid_arr = uid_arr[:n]
    iid_arr = iid_arr[:n]
    ts_arr  = ts_arr[:n]
    bhv_arr = bhv_arr[:n]
    print(f"Total rows loaded: {n:,}  (skipped malformed: {skipped_parse:,})", flush=True)

    # ------------------------------------------------------------------
    # Sanity report: behavior distribution over all loaded interactions.
    # This is the check that the BEHAVIOR_MAP strings actually match the raw
    # data — if any class is 0 here, the map keys are wrong.
    # ------------------------------------------------------------------
    print("\nBehavior distribution over loaded interactions:", flush=True)
    bhv_hist = np.bincount(bhv_arr.astype(np.int64), minlength=len(BEHAVIOR_ID_NAMES))
    for bid in sorted(BEHAVIOR_ID_NAMES):
        cnt = int(bhv_hist[bid])
        pct = cnt / n * 100 if n else 0
        flag = "  <-- EMPTY, map key likely wrong!" if cnt == 0 else ""
        print(f"  id={bid} {BEHAVIOR_ID_NAMES[bid]:<14}: {cnt:>14,}  ({pct:6.2f}%){flag}")
    print(f"\n  Skipped unknown-behavior rows: {skipped_unknown_bhv:,}")
    if unknown_bhv_strings:
        print("  Unrecognised behavior strings encountered:")
        for s, c in sorted(unknown_bhv_strings.items(), key=lambda kv: -kv[1]):
            print(f"    {repr(s):<20}: {c:,}")
    else:
        print("  (all behavior strings recognised — BEHAVIOR_MAP is correct)")

    # ------------------------------------------------------------------
    # Phase 2: Sort by (user_id, timestamp)
    # ------------------------------------------------------------------
    print("\nPhase 2: Sorting by (user_id, timestamp)...", flush=True)
    order   = np.lexsort((ts_arr, uid_arr))
    uid_arr = uid_arr[order]
    iid_arr = iid_arr[order]
    bhv_arr = bhv_arr[order]
    del order, ts_arr
    print("  Done.", flush=True)

    # ------------------------------------------------------------------
    # Phase 3: Group by user, (optionally) rebalance TRAINING, write TFRecords
    # ------------------------------------------------------------------
    print("\nPhase 3: Grouping users and writing TFRecord files...", flush=True)
    output_dir = Path(args.output_dir)
    SPLITS = ("training", "evaluation", "testing")
    for s in SPLITS:
        (output_dir / s).mkdir(parents=True, exist_ok=True)

    PV_ID = 0  # pv/click behavior id

    # --- Vectorised per-user table over the (already user-sorted) arrays ---
    boundaries = np.flatnonzero(np.diff(uid_arr)) + 1
    starts  = np.concatenate(([0], boundaries)).astype(np.int64)
    ends    = np.concatenate((boundaries, [n])).astype(np.int64)
    lengths = (ends - starts).astype(np.int64)
    u_uid   = uid_arr[starts]
    num_users = len(starts)

    # pv count per user via prefix sums
    pv_cumsum = np.concatenate(([0], np.cumsum((bhv_arr == PV_ID).astype(np.int64))))
    u_pv = pv_cumsum[ends] - pv_cumsum[starts]

    # split per user: bucket = (uid % 100) / 100  -> 0=training, 1=evaluation, 2=testing
    train_thresh = args.train_frac
    val_thresh   = args.train_frac + args.val_frac
    bucket = (u_uid % 100) / 100.0
    u_split = np.where(bucket < train_thresh, 0,
                       np.where(bucket < val_thresh, 1, 2)).astype(np.int8)

    # Natural length filter (applied to every split, every mode)
    len_ok = (lengths >= args.min_seq_len) & (lengths <= args.max_seq_len)
    users_skip = int((~len_ok).sum())
    is_train = (u_split == 0) & len_ok

    # ------------------------------------------------------------------
    # Calibration — TRAINING split only
    #   keep_user[g]  : whether user g is written at all (user_drop may flip to False)
    #   pv_keep_prob  : fraction of pv events kept per training seq (event_downsample)
    #   n_train_ref   : MODE-INDEPENDENT reference training size used to size val/test.
    #                   Always the user_drop greedy keep-count, so the val/test sets are
    #                   IDENTICAL whether you pick user_drop or event_downsample.
    # ------------------------------------------------------------------
    keep_user = len_ok.copy()
    pv_keep_prob = 1.0
    n_train_ref = int(is_train.sum())   # default: all natural (length-ok) training users

    if args.augment_mode != "none":
        train_pv_total  = int(u_pv[is_train].sum())
        train_all_total = int(lengths[is_train].sum())
        p0 = train_pv_total / train_all_total if train_all_total else 0.0
        t  = args.target_pv_ratio
        print(f"\n[rebalance] training natural pv ratio p0 = {p0:.4f}  (target t = {t:.3f})")

        if t >= p0:
            print("[rebalance] target >= natural pv ratio; nothing to do.")
        else:
            # --- user_drop greedy keep set: computed in BOTH modes ---
            # Drop the most pv-dominated training users first (highest pv-fraction ->
            # least non-pv yield per interaction) until the remaining training pv share
            # reaches the target. The kept count `n_train_ref` is mode-independent and
            # sizes val/test, guaranteeing an identical eval set across both modes.
            tr_idx = np.flatnonzero(is_train)
            pv_frac = u_pv[tr_idx] / lengths[tr_idx]
            order_desc = tr_idx[np.argsort(-pv_frac)]
            pv_run, all_run, dropped = train_pv_total, train_all_total, 0
            drop_list = []
            for g in order_desc:
                if all_run > 0 and (pv_run / all_run) <= t:
                    break
                drop_list.append(g)
                pv_run  -= int(u_pv[g])
                all_run -= int(lengths[g])
                dropped += 1
            n_train_ref = int(tr_idx.size) - dropped

            if args.augment_mode == "user_drop":
                if drop_list:
                    keep_user[np.asarray(drop_list, dtype=np.int64)] = False
                achieved = pv_run / all_run if all_run else 0.0
                print(f"[rebalance] user_drop: dropped {dropped:,} pv-dominated training "
                      f"users; training pv ratio now ~{achieved:.4f} over {all_run:,} "
                      f"interactions ({n_train_ref:,} training users kept).")
            elif args.augment_mode == "event_downsample":
                # keep fraction k of pv events so that k*P / (k*P + Q) = t
                pv_keep_prob = t * (1.0 - p0) / (p0 * (1.0 - t))
                pv_keep_prob = max(0.0, min(1.0, pv_keep_prob))
                print(f"[rebalance] event_downsample: keep {pv_keep_prob*100:.2f}% of pv "
                      f"events (drop {(1-pv_keep_prob)*100:.2f}%) within training sequences; "
                      f"all cart/collect/alipay kept.")
                print(f"[rebalance] val/test sized to the user_drop reference "
                      f"({n_train_ref:,} train users) -> identical eval set across modes.")

    # ------------------------------------------------------------------
    # Write pass
    #
    # Training is written first, then validation/testing are RANDOMLY subsampled
    # (distribution-preserving — they stay at the natural ~93.6% pv) down to the
    # original split proportion, sized from the MODE-INDEPENDENT reference
    # `n_train_ref` so the eval set is identical across user_drop / event_downsample:
    #     target_val  = n_train_ref * (val_frac  / train_frac)
    #     target_test = n_train_ref * (test_frac / train_frac)
    # ------------------------------------------------------------------
    buffers     = {s: [] for s in SPLITS}
    file_counts = {s: 0  for s in SPLITS}
    users_done  = {s: 0  for s in SPLITS}
    # Behavior histogram over WRITTEN interactions (post-rebalancing), per split.
    written_bhv = {s: np.zeros(len(BEHAVIOR_ID_NAMES), dtype=np.int64) for s in SPLITS}
    users_downsample_skip = 0  # training users that fell below min_seq_len after thinning

    tf_opts = tf.io.TFRecordOptions(compression_type="GZIP")

    def flush(split: str):
        if not buffers[split]:
            return
        idx      = file_counts[split]
        out_path = str(output_dir / split / f"data_{idx:05d}.tfrecord.gz")
        with tf.io.TFRecordWriter(out_path, options=tf_opts) as writer:
            for uid, seq, bhv in buffers[split]:
                feat = {
                    "user_id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[uid])
                    ),
                    "sequence_data": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=seq)
                    ),
                    "behavior_data": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=bhv)
                    ),
                }
                writer.write(
                    tf.train.Example(
                        features=tf.train.Features(feature=feat)
                    ).SerializeToString()
                )
        file_counts[split] += 1
        buffers[split] = []

    def write_user(g: int, split: str) -> None:
        nonlocal users_downsample_skip
        s_, e_ = int(starts[g]), int(ends[g])
        seq = iid_arr[s_:e_]
        bhv = bhv_arr[s_:e_]
        # event_downsample: thin pv events (training split only)
        if (args.augment_mode == "event_downsample" and split == "training"
                and pv_keep_prob < 1.0):
            keep_mask = (bhv != PV_ID) | (np.random.random(len(bhv)) < pv_keep_prob)
            seq = seq[keep_mask]
            bhv = bhv[keep_mask]
            if len(seq) < args.min_seq_len:
                users_downsample_skip += 1
                return
        buffers[split].append((int(u_uid[g]), seq.tolist(), bhv.tolist()))
        users_done[split] += 1
        written_bhv[split] += np.bincount(bhv, minlength=len(BEHAVIOR_ID_NAMES))
        if len(buffers[split]) >= args.users_per_file:
            flush(split)

    # --- (3a) Training split first ---
    for g in np.flatnonzero((u_split == 0) & keep_user):
        write_user(int(g), "training")
    flush("training")
    n_train_written = users_done["training"]

    # --- (3b) Size val/test from the mode-independent reference, then subsample ---
    # Using n_train_ref (not n_train_written) keeps the val/test selection identical
    # across user_drop and event_downsample.
    test_frac = max(0.0, 1.0 - args.train_frac - args.val_frac)
    split_targets = {
        "evaluation": int(round(n_train_ref * args.val_frac / args.train_frac)),
        "testing":    int(round(n_train_ref * test_frac     / args.train_frac)),
    }
    rng_sub = np.random.default_rng(args.seed + 1)  # separate stream for subsampling
    eval_test_subsampled = {}
    for split, sidx in (("evaluation", 1), ("testing", 2)):
        elig = np.flatnonzero((u_split == sidx) & keep_user)
        target = split_targets[split]
        if args.augment_mode != "none" and 0 <= target < len(elig):
            chosen = np.sort(rng_sub.choice(elig, size=target, replace=False))
            eval_test_subsampled[split] = len(elig) - target
        else:
            chosen = elig
            eval_test_subsampled[split] = 0
        for g in chosen:
            write_user(int(g), split)
        flush(split)

    print("\nDone!")
    print(f"  Users — train: {users_done['training']:,}  "
          f"val: {users_done['evaluation']:,}  "
          f"test: {users_done['testing']:,}  "
          f"skipped (len not in [{args.min_seq_len}, {args.max_seq_len}]): {users_skip:,}")
    if args.augment_mode == "user_drop":
        dropped_total = int((is_train & ~keep_user).sum())
        print(f"  Rebalance (user_drop) — dropped {dropped_total:,} pv-dominated training users")
    elif args.augment_mode == "event_downsample":
        print(f"  Rebalance (event_downsample) — pv keep prob {pv_keep_prob:.4f}; "
              f"training users dropped post-thinning (<min_seq_len): {users_downsample_skip:,}")
    if args.augment_mode != "none":
        print(f"  Val/test proportional subsample (natural distribution kept) — "
              f"val dropped: {eval_test_subsampled.get('evaluation', 0):,}  "
              f"test dropped: {eval_test_subsampled.get('testing', 0):,}")
        print(f"  Split user ratio now ~ {args.train_frac:.2f} : {args.val_frac:.2f} : "
              f"{test_frac:.2f}  "
              f"(train {users_done['training']:,} / val {users_done['evaluation']:,} / "
              f"test {users_done['testing']:,})")
    print(f"  Files — train: {file_counts['training']:,}  "
          f"val: {file_counts['evaluation']:,}  "
          f"test: {file_counts['testing']:,}")
    print(f"  Output: {output_dir}")

    # ------------------------------------------------------------------
    # Behavior ratio AFTER transformation (over interactions actually written)
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("Behavior ratio AFTER transformation (written interactions)")
    print("=" * 62)
    total_written = np.zeros(len(BEHAVIOR_ID_NAMES), dtype=np.int64)
    for s in SPLITS:
        total_written += written_bhv[s]
    grand = int(total_written.sum())
    print(f"\n  GLOBAL  (total written interactions: {grand:,})")
    for bid in sorted(BEHAVIOR_ID_NAMES):
        cnt = int(total_written[bid])
        pct = cnt / grand * 100 if grand else 0
        print(f"    id={bid} {BEHAVIOR_ID_NAMES[bid]:<14}: {cnt:>14,}  ({pct:6.2f}%)")
    for s in SPLITS:
        h = written_bhv[s]
        tot = int(h.sum())
        print(f"\n  [{s}]  (total: {tot:,})")
        for bid in sorted(BEHAVIOR_ID_NAMES):
            cnt = int(h[bid])
            pct = cnt / tot * 100 if tot else 0
            print(f"    id={bid} {BEHAVIOR_ID_NAMES[bid]:<14}: {cnt:>14,}  ({pct:6.2f}%)")
    print("=" * 62)


if __name__ == "__main__":
    main()