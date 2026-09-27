"""
Production Inference Pipeline: Candidate Scoring, Results Generation & Format Validation.

Takes:
  - output/candidate_pairs.tsv (from Step 1)
  - models/final_matcher.pkl (from Step 4 Tournament)
Outputs:
  - output/matching_results.tsv (Official Leaderboard Submission File)
Validates:
  - Invokes utils/validate_submission.py to guarantee PASS (Exit 0)
"""

import os
import sys
import time
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd

try:
    from src.features import EntityProfile, compute_pairwise_features
except ImportError:
    from features import EntityProfile, compute_pairwise_features

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_model_bundle(model_path: str):
    print(f"Loading trained matcher bundle from {model_path}...")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def predict_pair_probs(features_matrix: np.ndarray, bundle: dict, mode: str = "ens2") -> np.ndarray:
    """Predict match probabilities using individual model or ensemble."""
    if len(features_matrix) == 0:
        return np.array([])

    if mode == "lgbm":
        return bundle["lgbm"].predict_proba(features_matrix)[:, 1]
    elif mode == "ens2":
        p_lgb = bundle["lgbm"].predict_proba(features_matrix)[:, 1]
        p_cat = bundle["catboost"].predict_proba(features_matrix)[:, 1]
        return 0.5 * p_lgb + 0.5 * p_cat
    elif mode == "ens3":
        p_lgb = bundle["lgbm"].predict_proba(features_matrix)[:, 1]
        p_cat = bundle["catboost"].predict_proba(features_matrix)[:, 1]
        p_xgb = bundle["xgboost"].predict_proba(features_matrix)[:, 1]
        return (p_lgb + p_cat + p_xgb) / 3.0
    else:
        return bundle["lgbm"].predict_proba(features_matrix)[:, 1]


def run_inference(
    candidates_path: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    model_path: str,
    output_path: str,
    mode: str = "ens2",
    threshold_override: float = None,
    sample_limit: int = None
):
    start_time = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=" * 85)
    print("              PRODUCTION ENTITY RESOLUTION INFERENCE PIPELINE")
    print("=" * 85)
    print(f"Candidate File:  {candidates_path}")
    print(f"Source 1 File:   {s1_path}")
    print(f"Output File:     {output_path}")
    print(f"Inference Mode:  {mode.upper()}")

    # 1. Load Model Bundle
    bundle = load_model_bundle(model_path)
    if threshold_override is not None:
        threshold = threshold_override
    elif mode == "ens2":
        threshold = bundle.get("optimal_threshold_ens2", 0.50)
    elif mode == "ens3":
        threshold = bundle.get("optimal_threshold_ens3", 0.50)
    else:
        threshold = bundle.get("optimal_threshold_lgbm", 0.50)

    print(f"Active Decision Threshold: {threshold:.2f} (Optimized for F_0.5 Precision)")

    # 2. Read Candidates Mapping
    print("\nStep 1: Reading candidate pairs...")
    t0 = time.time()
    cand_map = {}
    needed_s23_ids = set()

    with open(candidates_path, "r", encoding="utf-8") as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1].strip() else []
            cand_map[s1_id] = cands
            for c in cands:
                needed_s23_ids.add(c)
            if sample_limit and len(cand_map) >= sample_limit:
                break

    print(f"  Loaded candidates for {len(cand_map):,} S1 entities ({len(needed_s23_ids):,} unique S2/S3 IDs) in {time.time() - t0:.2f}s")

    # 3. Load S2 and S3 Records (store raw strings to keep RAM footprint minimal)
    print("\nStep 2: Ingesting S2 and S3 candidate records (lightweight storage)...")
    t0 = time.time()
    s23_records = {}
    for p in [s2_path, s3_path]:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                eid = parts[0]
                if eid in needed_s23_ids:
                    s23_records[eid] = (parts[1], parts[2], parts[3])

    print(f"  Ingested {len(s23_records):,} candidate records in {time.time() - t0:.2f}s")

    # 4. Stream S1, Score Candidates, and Write matching_results.tsv
    print("\nStep 3: Scoring candidates and writing matching_results.tsv...")
    t0 = time.time()
    total_processed = 0
    total_matches = 0
    singletons_count = 0

    with open(s1_path, "r", encoding="utf-8", errors="replace") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        in_f.readline()  # Skip header

        for line in in_f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[0]
            if eid not in cand_map:
                continue

            total_processed += 1
            b_name = parts[1] if len(parts) > 1 else ""
            b_addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""

            cand_ids = cand_map[eid]
            if not cand_ids:
                singletons_count += 1
                out_f.write(f"{eid}\t\n")
                continue

            # Build S1 profile on the fly
            p1 = EntityProfile(eid, b_name, b_addr, country)

            # Compute features for candidate batch on the fly
            feats_batch = []
            valid_cand_ids = []

            for cid in cand_ids:
                if cid in s23_records:
                    rec = s23_records[cid]
                    p2 = EntityProfile(cid, rec[0], rec[1], rec[2])
                    feats = compute_pairwise_features(p1, p2)
                    feats_batch.append(feats)
                    valid_cand_ids.append(cid)

            if not feats_batch:
                singletons_count += 1
                out_f.write(f"{eid}\t\n")
                continue

            # Score with model/ensemble
            probs = predict_pair_probs(np.array(feats_batch), bundle, mode=mode)
            matched = [valid_cand_ids[i] for i in range(len(probs)) if probs[i] >= threshold]

            if not matched:
                singletons_count += 1
                out_f.write(f"{eid}\t\n")
            else:
                total_matches += len(matched)
                out_f.write(f"{eid}\t{','.join(matched)}\n")

            if total_processed % 50000 == 0:
                print(f"  Processed {total_processed:,} S1 entities... (Matches: {total_matches:,})")

    elapsed = time.time() - t0
    print("\n" + "=" * 85)
    print("                     INFERENCE SUMMARY REPORT")
    print("=" * 85)
    print(f"Total S1 Entities Evaluated:  {total_processed:,}")
    print(f"Total Resolved Matches:       {total_matches:,}")
    print(f"Singletons (Zero Matches):    {singletons_count:,} ({singletons_count / total_processed * 100:.1f}%)")
    print(f"Avg Matches / Entity:         {total_matches / total_processed:.2f}")
    print(f"Inference Runtime:            {elapsed:.2f}s")
    print(f"Saved Matching File:          {output_path}")
    print("=" * 85)

    # 5. Run Submission Validator
    print("\nStep 4: Running official submission validator...")
    validator_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", output_path,
        "--candidate", candidates_path,
        "--test-dir", os.path.dirname(s1_path)
    ]
    val_proc = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr:
        print(val_proc.stderr)

    if val_proc.returncode == 0:
        print("\n [VALIDATOR PASSED] Output files are 100% compliant and ready to submit!")
    else:
        print("\n [VALIDATOR ALERT] Check above messages for formatting requirements.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Entity Matching Inference")
    parser.add_argument("--candidates", type=str, default="output/candidate_pairs.tsv", help="Candidate TSV path")
    parser.add_argument("--s1", type=str, default="dataset/test/test_source1.tsv", help="Source 1 TSV")
    parser.add_argument("--s2", type=str, default="dataset/test/test_source2.tsv", help="Source 2 TSV")
    parser.add_argument("--s3", type=str, default="dataset/test/test_source3.tsv", help="Source 3 TSV")
    parser.add_argument("--model", type=str, default="models/final_matcher.pkl", help="Trained model bundle path")
    parser.add_argument("--output", type=str, default="output/matching_results.tsv", help="Output matching results TSV")
    parser.add_argument("--mode", type=str, default="ens2", choices=["lgbm", "ens2", "ens3"], help="Inference model mode")
    parser.add_argument("--threshold", type=float, default=None, help="Optional manual threshold override")
    parser.add_argument("--sample", type=int, default=None, help="Process only N records for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        candidates_path=args.candidates,
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        model_path=args.model,
        output_path=args.output,
        mode=args.mode,
        threshold_override=args.threshold,
        sample_limit=args.sample
    )
