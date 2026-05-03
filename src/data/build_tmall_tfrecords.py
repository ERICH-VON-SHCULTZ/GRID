"""
Convert Tmall valid_data interaction logs into GZIP TFRecord files for TIGER T5 training.

Each TFRecord row = one user's full chronological interaction sequence:
  - sequence_data : int64 list  (item IDs, sorted by timestamp)
  - user_id       : int64 scalar

Users are split deterministically (by user_id % 100) into training / evaluation / testing.

Memory usage: 3 numpy int32/int64 arrays for all interactions, ~15 GB peak.

Usage:
    python src/data/build_tmall_tfrecords.py \
        --valid_data_dir /scratch/yw8866/rec-tmall/valid_data \
        --output_dir     /scratch/yw8866/rec-tmall/tfrecords \
        --min_seq_len    5 \
        --users_per_file 5000 \
        --train_frac     0.8 \
        --val_frac       0.1
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

SEP = "\x01"
LOG_FILES = [
    "tianchi_2014002_rec_tmall_log_parta.txt",
    "tianchi_2014002_rec_tmall_log_partb.txt",
    "tianchi_2014002_rec_tmall_log_partc.txt",
]


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
    parser.add_argument("--behavior_types", type=str, default="",
                        help="Comma-separated behavior types to keep, e.g. 'alipay,cart'. "
                             "Empty string = keep all (default)")
    args = parser.parse_args()

    keep_behaviors = (
        set(b.strip() for b in args.behavior_types.split(",") if b.strip())
        if args.behavior_types else None
    )
    if keep_behaviors:
        print(f"Keeping only behavior types: {keep_behaviors}")
    else:
        print("Keeping all behavior types.")

    # ------------------------------------------------------------------
    # Phase 1: Read all log files into three compact numpy arrays
    # ------------------------------------------------------------------
    print("\nPhase 1: Reading interaction logs...", flush=True)
    MAX_ROWS = 1_200_000_000
    uid_arr  = np.empty(MAX_ROWS, dtype=np.int32)
    iid_arr  = np.empty(MAX_ROWS, dtype=np.int32)
    ts_arr   = np.empty(MAX_ROWS, dtype=np.int64)
    n = 0
    skipped_parse = 0

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
                    if keep_behaviors and btype not in keep_behaviors:
                        continue
                    item_id = int(parts[0].strip())
                    uid_str = parts[1].strip()
                    user_id = int(uid_str[1:] if uid_str.startswith("u") else uid_str)
                    ts      = parse_ts(parts[3])
                    uid_arr[n] = user_id
                    iid_arr[n] = item_id
                    ts_arr[n]  = ts
                    n += 1
                except (ValueError, IndexError):
                    skipped_parse += 1
        print(f"    Rows so far: {n:,}", flush=True)

    uid_arr = uid_arr[:n]
    iid_arr = iid_arr[:n]
    ts_arr  = ts_arr[:n]
    print(f"Total rows loaded: {n:,}  (skipped malformed: {skipped_parse:,})", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: Sort by (user_id, timestamp)
    # ------------------------------------------------------------------
    print("\nPhase 2: Sorting by (user_id, timestamp)...", flush=True)
    order   = np.lexsort((ts_arr, uid_arr))
    uid_arr = uid_arr[order]
    iid_arr = iid_arr[order]
    del order, ts_arr
    print("  Done.", flush=True)

    # ------------------------------------------------------------------
    # Phase 3: Stream through sorted array, write TFRecords split by user
    # ------------------------------------------------------------------
    print("\nPhase 3: Writing TFRecord files...", flush=True)
    output_dir = Path(args.output_dir)
    SPLITS = ("training", "evaluation", "testing")
    for s in SPLITS:
        (output_dir / s).mkdir(parents=True, exist_ok=True)

    train_thresh = args.train_frac
    val_thresh   = args.train_frac + args.val_frac

    def user_split(uid: int) -> str:
        bucket = (uid % 100) / 100.0
        if bucket < train_thresh:
            return "training"
        if bucket < val_thresh:
            return "evaluation"
        return "testing"

    buffers     = {s: [] for s in SPLITS}
    file_counts = {s: 0  for s in SPLITS}
    users_done  = {s: 0  for s in SPLITS}
    users_skip  = 0

    tf_opts = tf.io.TFRecordOptions(compression_type="GZIP")

    def flush(split: str):
        if not buffers[split]:
            return
        idx      = file_counts[split]
        out_path = str(output_dir / split / f"data_{idx:05d}.tfrecord.gz")
        with tf.io.TFRecordWriter(out_path, options=tf_opts) as writer:
            for uid, seq in buffers[split]:
                feat = {
                    "user_id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[uid])
                    ),
                    "sequence_data": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=seq)
                    ),
                }
                writer.write(
                    tf.train.Example(
                        features=tf.train.Features(feature=feat)
                    ).SerializeToString()
                )
        file_counts[split] += 1
        buffers[split] = []

    cur_uid = None
    cur_seq: list = []

    def commit(uid: int, seq: list):
        nonlocal users_skip
        if len(seq) < args.min_seq_len or len(seq) > args.max_seq_len:
            users_skip += 1
            return
        split = user_split(uid)
        buffers[split].append((uid, seq))
        users_done[split] += 1
        if len(buffers[split]) >= args.users_per_file:
            flush(split)

    for i in range(n):
        uid = int(uid_arr[i])
        iid = int(iid_arr[i])
        if uid != cur_uid:
            if cur_uid is not None:
                commit(cur_uid, cur_seq)
            cur_uid = uid
            cur_seq = [iid]
        else:
            cur_seq.append(iid)

    if cur_uid is not None:
        commit(cur_uid, cur_seq)

    for s in SPLITS:
        flush(s)

    print("\nDone!")
    print(f"  Users — train: {users_done['training']:,}  "
          f"val: {users_done['evaluation']:,}  "
          f"test: {users_done['testing']:,}  "
          f"skipped (seq not in [{args.min_seq_len}, {args.max_seq_len}]): {users_skip:,}")
    print(f"  Files — train: {file_counts['training']:,}  "
          f"val: {file_counts['evaluation']:,}  "
          f"test: {file_counts['testing']:,}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()