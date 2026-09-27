"""
Explore the business entity resolution dataset.
Run this FIRST before writing any pipeline code — you need to know
what you're actually dealing with (noise patterns, size, field quality).

Usage:
    python src/explore_data.py
(run from the student_resource/ root, or adjust the paths below)
"""

import os
import re
import sys
from collections import Counter

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"


def analyze_tsv(path, name):
    print(f"\n{'='*60}")
    print(f"{name}: {path}")
    print(f"{'='*60}")
    
    total_lines = 0
    header = []
    empty_counts = Counter()
    country_counts = Counter()
    sample_rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            print("Empty file!")
            return 0, []
        header = [col.strip() for col in header_line.split("\t")]
        country_col_idx = header.index("country") if "country" in header else None

        for line in f:
            total_lines += 1
            parts = line.rstrip("\r\n").split("\t")
            if total_lines <= 5:
                sample_rows.append(parts)
            for idx, col in enumerate(header):
                val = parts[idx].strip() if idx < len(parts) else ""
                if not val:
                    empty_counts[col] += 1
            if country_col_idx is not None and country_col_idx < len(parts):
                c = parts[country_col_idx].strip()
                if c:
                    country_counts[c] += 1

    print(f"Total Rows: {total_lines:,} | Columns: {len(header)} ({', '.join(header)})")
    print("\nSample rows (first 5):")
    for r in sample_rows:
        print("  " + " | ".join(r[:4]))

    print("\nEmpty counts per column:")
    for col in header:
        print(f"  {col}: {empty_counts[col]:,} empty")

    if country_counts:
        print("\nCountry value counts:")
        for c, count in country_counts.most_common():
            print(f"  {c}: {count:,} ({count / total_lines * 100:.1f}%)")

    return total_lines, header


def analyze_ground_truth(gt_path):
    print(f"\n{'='*60}")
    print(f"GROUND TRUTH ANALYSIS: {gt_path}")
    print(f"{'='*60}")

    total_s1 = 0
    singletons = 0
    match_distribution = Counter()
    total_matches = 0
    s2_count = 0
    s3_count = 0
    sample_pairs = []

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            total_s1 += 1
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            if not matched_str:
                singletons += 1
                match_distribution[0] += 1
            else:
                ids = [m.strip() for m in matched_str.split(",") if m.strip()]
                num_m = len(ids)
                match_distribution[num_m] += 1
                total_matches += num_m
                for m in ids:
                    if m.startswith("S2-"):
                        s2_count += 1
                    elif m.startswith("S3-"):
                        s3_count += 1
                if len(sample_pairs) < 8:
                    sample_pairs.append((s1_id, ids))

    print(f"Total S1 entities in ground truth: {total_s1:,}")
    print(f"Singletons (no match): {singletons:,} ({singletons / total_s1 * 100:.1f}%)")
    print(f"Mean matches per entity: {total_matches / total_s1:.2f}")
    print(f"Total matched IDs: {total_matches:,} (S2: {s2_count:,}, S3: {s3_count:,})")
    print("\nMatches per S1 entity distribution:")
    for num_matches in sorted(match_distribution.keys())[:10]:
        print(f"  {num_matches} matches: {match_distribution[num_matches]:,} entities")

    return sample_pairs


def show_sample_pairs(sample_pairs, s1_path, s2_path, s3_path):
    print(f"\n{'='*60}")
    print("SAMPLE MATCHED PAIRS (Detailed Noise Inspection)")
    print(f"{'='*60}")

    target_s1_ids = {pair[0] for pair in sample_pairs}
    target_s23_ids = {mid for pair in sample_pairs for mid in pair[1]}

    records = {}
    for p in [s1_path, s2_path, s3_path]:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts and (parts[0] in target_s1_ids or parts[0] in target_s23_ids):
                    records[parts[0]] = parts

    for s1_id, matched_ids in sample_pairs:
        s1_row = records.get(s1_id, [s1_id, "N/A", "N/A", "N/A"])
        print(f"\n--- {s1_id} ---")
        print(f"  NAME: {s1_row[1] if len(s1_row) > 1 else ''}")
        print(f"  ADDR: {s1_row[2] if len(s1_row) > 2 else ''}")
        print(f"  CTRY: {s1_row[3] if len(s1_row) > 3 else ''}")
        for mid in matched_ids:
            m_row = records.get(mid, [mid, "N/A", "N/A", "N/A"])
            print(f"   -> {mid}")
            print(f"      NAME: {m_row[1] if len(m_row) > 1 else ''}")
            print(f"      ADDR: {m_row[2] if len(m_row) > 2 else ''}")
            print(f"      CTRY: {m_row[3] if len(m_row) > 3 else ''}")


def main():
    # 1. Analyze Train Files
    analyze_tsv(f"{TRAIN_DIR}/train_source1.tsv", "TRAIN Source 1 (reference)")
    analyze_tsv(f"{TRAIN_DIR}/train_source2.tsv", "TRAIN Source 2")
    analyze_tsv(f"{TRAIN_DIR}/train_source3.tsv", "TRAIN Source 3")

    # 2. Analyze Ground Truth
    sample_pairs = analyze_ground_truth(f"{TRAIN_DIR}/train_ground_truth.tsv")

    # 3. Show Sample Pairs
    show_sample_pairs(
        sample_pairs,
        f"{TRAIN_DIR}/train_source1.tsv",
        f"{TRAIN_DIR}/train_source2.tsv",
        f"{TRAIN_DIR}/train_source3.tsv"
    )

    # 4. Analyze Test Files
    print(f"\n{'='*60}")
    print("TEST SET SIZES (Watch for France in test!)")
    print(f"{'='*60}")
    analyze_tsv(f"{TEST_DIR}/test_source1.tsv", "TEST Source 1")
    analyze_tsv(f"{TEST_DIR}/test_source2.tsv", "TEST Source 2")
    analyze_tsv(f"{TEST_DIR}/test_source3.tsv", "TEST Source 3")


if __name__ == "__main__":
    main()