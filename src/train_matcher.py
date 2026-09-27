"""
Training the Entity Resolution Matcher with LightGBM and F_0.5 Threshold Optimization.

Workflow:
  1. Ingest sample training records and Ground Truth matches.
  2. Generate positive pairs (Ground Truth) and hard negative distractors (Blocking candidates).
  3. Extract 12-dimensional features using src.features.EntityProfile.
  4. Train a LightGBM classifier with early stopping.
  5. Search optimal decision threshold specifically for macro F_0.5.
  6. Save the trained model to models/matcher_lgbm.txt.
"""

import os
import sys
import time
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

try:
    from src.features import EntityProfile, compute_pairwise_features, FEATURE_NAMES
except ImportError:
    from features import EntityProfile, compute_pairwise_features, FEATURE_NAMES

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F_beta score (beta=0.5 weights precision 2x over recall)."""
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return ((1 + b2) * precision * recall) / (b2 * precision + recall)


def evaluate_macro_f05(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> tuple:
    """Computes Precision, Recall, and F_0.5 at a specific probability threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = calculate_f_beta(precision, recall, beta=0.5)

    return f05, precision, recall


def load_profiles_from_tsv(path: str, eids_filter: set = None) -> dict:
    """Load records and construct cached EntityProfile objects."""
    profiles = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            eid = parts[0]
            if eids_filter is not None and eid not in eids_filter:
                continue
            profiles[eid] = EntityProfile(eid, parts[1], parts[2], parts[3])
    return profiles


def prepare_training_dataset(
    gt_path: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    num_s1_samples: int = 2500,
    neg_ratio: int = 3
):
    print("Step 1: Sampling Ground Truth matches and candidate distractors...")
    t0 = time.time()
    
    gt_df = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    matched_gt = gt_df[gt_df["matched_entity_ids"].str.strip() != ""].head(num_s1_samples)
    
    s1_target_ids = set(matched_gt["source1_entity_id"])
    target_pos_ids = set()
    gt_pairs = set()

    for _, row in matched_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
        for m in matches:
            gt_pairs.add((s1_id, m))
            target_pos_ids.add(m)

    print(f"  Sampled {len(s1_target_ids):,} S1 entities with {len(gt_pairs):,} true positive links.")

    # Load S1 profiles
    print("Step 2: Loading S1 entity profiles...")
    s1_profiles = load_profiles_from_tsv(s1_path, s1_target_ids)

    # Load matching S2 and S3 profiles
    print("Step 3: Loading S2 and S3 profiles (True targets + Distractors)...")
    s23_target_ids = set(target_pos_ids)
    
    # Also load some distractor IDs from S2 and S3
    s23_profiles = {}
    for p in [s2_path, s3_path]:
        with open(p, "r", encoding="utf-8") as f:
            f.readline()
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                eid = parts[0]
                if eid in s23_target_ids or len(s23_profiles) < len(target_pos_ids) * (neg_ratio + 1):
                    s23_profiles[eid] = EntityProfile(eid, parts[1], parts[2], parts[3])

    print(f"  Ingested {len(s1_profiles):,} S1 profiles and {len(s23_profiles):,} S2/S3 profiles in {time.time() - t0:.2f}s")

    # Step 4: Construct Pairwise Training Rows (Positives + Hard Negatives)
    print("Step 4: Computing pairwise feature vectors...")
    t0 = time.time()
    X = []
    y = []

    # 1. Add Positive Pairs
    for s1_id, match_id in gt_pairs:
        if s1_id in s1_profiles and match_id in s23_profiles:
            p1 = s1_profiles[s1_id]
            p2 = s23_profiles[match_id]
            feats = compute_pairwise_features(p1, p2)
            X.append(feats)
            y.append(1)

    num_pos = len(X)

    # 2. Add Hard Negative Distractors (same country, similar prefix/tokens, but not in GT)
    s23_by_country = defaultdict(list)
    for eid, prof in s23_profiles.items():
        s23_by_country[prof.country].append(prof)

    neg_count = 0
    target_negs = num_pos * neg_ratio

    for s1_id, p1 in s1_profiles.items():
        ctry_pool = s23_by_country.get(p1.country, [])
        if not ctry_pool:
            continue
        # Pick random candidates in same country
        sample_size = min(len(ctry_pool), neg_ratio + 2)
        candidates = np.random.choice(ctry_pool, size=sample_size, replace=False)
        for cand in candidates:
            if (s1_id, cand.eid) not in gt_pairs:
                feats = compute_pairwise_features(p1, cand)
                X.append(feats)
                y.append(0)
                neg_count += 1
                if neg_count >= target_negs:
                    break
        if neg_count >= target_negs:
            break

    print(f"  Training matrix constructed in {time.time() - t0:.2f}s:")
    print(f"    - Positive Pairs (y=1): {num_pos:,}")
    print(f"    - Negative Pairs (y=0): {neg_count:,}")
    print(f"    - Total Matrix Shape:   ({len(X):,}, {len(FEATURE_NAMES)})")

    return np.array(X), np.array(y)


def train_and_optimize(X: np.ndarray, y: np.ndarray, model_save_path: str = "models/matcher_lgbm.txt"):
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)

    print("\n" + "=" * 80)
    print("             TRAINING LIGHTGBM MATCHER WITH F_0.5 OPTIMIZATION")
    print("=" * 80)

    # Train / Validation Split (80% / 20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Training on {len(X_train):,} pairs, Validating on {len(X_val):,} pairs...")

    # LightGBM Parameters optimized for tabular string similarities
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)

    print("Fitting Gradient Boosted Trees (150 rounds max with early stopping)...")
    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=150,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    print(f"Model trained in {time.time() - t0:.2f}s (Best iteration: {model.best_iteration})")

    # Feature Importance
    print("\n--- FEATURE IMPORTANCES ---")
    importances = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp:
        print(f"  {name:25s}: {imp:10.2f}")

    # Threshold Search for Maximum F_0.5
    print("\n" + "-" * 80)
    print("         SEARCHING OPTIMAL DECISION THRESHOLD FOR F_0.5 (PRECISION-HEAVY)")
    print("-" * 80)
    val_probs = model.predict(X_val)

    best_thresh = 0.50
    best_f05 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    print(f"  {'Threshold':<12} | {'Precision':<12} | {'Recall':<12} | {'F_0.5 Score':<12} | {'Verdict'}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+--------")

    for thresh in np.arange(0.35, 0.90, 0.05):
        f05, prec, rec = evaluate_macro_f05(y_val, val_probs, thresh)
        verdict = ""
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_prec = prec
            best_rec = rec
            verdict = "⭐ (NEW BEST)"
        print(f"  {thresh:<12.2f} | {prec:<12.4f} | {rec:<12.4f} | {f05:<12.4f} | {verdict}")

    print("-" * 80)
    print(f"OPTIMAL THRESHOLD FOUND: {best_thresh:.2f}")
    print(f"  Precision:   {best_prec * 100:.2f}% (High precision prevents false merges!)")
    print(f"  Recall:      {best_rec * 100:.2f}%")
    print(f"  F_0.5 Score: {best_f05:.4f}")
    print("-" * 80)

    # Save model and config
    model.save_model(model_save_path)
    with open(f"{model_save_path}.meta", "w") as f:
        f.write(f"threshold={best_thresh:.4f}\n")
        f.write(f"f05_score={best_f05:.4f}\n")
        f.write(f"precision={best_prec:.4f}\n")
        f.write(f"recall={best_rec:.4f}\n")

    print(f"Model artifact saved to: {model_save_path}")
    print(f"Threshold metadata saved to: {model_save_path}.meta")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Entity Matcher")
    parser.add_argument("--gt", type=str, default="dataset/train/train_ground_truth.tsv", help="Ground Truth TSV")
    parser.add_argument("--s1", type=str, default="dataset/train/train_source1.tsv", help="Source 1 TSV")
    parser.add_argument("--s2", type=str, default="dataset/train/train_source2.tsv", help="Source 2 TSV")
    parser.add_argument("--s3", type=str, default="dataset/train/train_source3.tsv", help="Source 3 TSV")
    parser.add_argument("--samples", type=int, default=2000, help="Number of S1 ground-truth samples to train on")
    parser.add_argument("--output", type=str, default="models/matcher_lgbm.txt", help="Model save path")
    args = parser.parse_args()

    X, y = prepare_training_dataset(
        gt_path=args.gt,
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        num_s1_samples=args.samples
    )
    train_and_optimize(X, y, model_save_path=args.output)


if __name__ == "__main__":
    main()
