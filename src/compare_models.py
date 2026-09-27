"""
Model Tournament, Threshold Optimization, and Ensembling Benchmark for Entity Resolution.

Workflow:
  1. Ingest Ground Truth positive matches and hard negative candidate distractors.
  2. Perform GroupKFold (split strictly by source1_entity_id to PREVENT data leakage).
  3. Train and benchmark 5 diverse models on identical 12 features:
     - Logistic Regression (Linear baseline)
     - Random Forest (Bagging ensemble)
     - XGBoost (Extreme Gradient Boosting)
     - CatBoost (Symmetric tree boosting)
     - LightGBM (Leaf-wise gradient boosting)
  4. Perform threshold grid-search (0.40 to 0.85) to find optimal F_0.5 per model.
  5. Build Soft-Voting Ensemble (averaging top models) and tune its threshold.
  6. Output final comparative report and save winning model/ensemble to models/final_matcher.pkl.
"""

import os
import sys
import time
import pickle
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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


def evaluate_f05_at_threshold(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> tuple:
    """Computes Precision, Recall, and F_0.5 at a given probability threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = calculate_f_beta(precision, recall, beta=0.5)

    return f05, precision, recall


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray, thresholds=np.arange(0.40, 0.86, 0.05)):
    """Grid search for threshold maximizing F_0.5."""
    best_f05 = 0.0
    best_thresh = 0.50
    best_prec = 0.0
    best_rec = 0.0

    for thresh in thresholds:
        f05, prec, rec = evaluate_f05_at_threshold(y_true, y_probs, thresh)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_prec = prec
            best_rec = rec

    return best_thresh, best_f05, best_prec, best_rec


def load_profiles_from_tsv(path: str, eids_filter: set = None) -> dict:
    profiles = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            eid = parts[0]
            if eids_filter is not None and eid not in eids_filter:
                continue
            profiles[eid] = EntityProfile(eid, parts[1], parts[2], parts[3])
    return profiles


def prepare_group_dataset(
    gt_path: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    num_s1_samples: int = 1500,
    neg_ratio: int = 3
):
    print("Step 1: Sampling Ground Truth matches and negative distractors...")
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
    s1_profiles = load_profiles_from_tsv(s1_path, s1_target_ids)

    # Load S2 and S3 profiles
    s23_target_ids = set(target_pos_ids)
    s23_profiles = {}
    for p in [s2_path, s3_path]:
        with open(p, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                eid = parts[0]
                if eid in s23_target_ids or len(s23_profiles) < len(target_pos_ids) * (neg_ratio + 1):
                    s23_profiles[eid] = EntityProfile(eid, parts[1], parts[2], parts[3])

    print(f"  Profiles ready in {time.time() - t0:.2f}s: S1={len(s1_profiles):,}, S2/S3={len(s23_profiles):,}")

    # Build features with groups (s1_id is group key!)
    X = []
    y = []
    groups = []

    # 1. Positives
    for s1_id, match_id in gt_pairs:
        if s1_id in s1_profiles and match_id in s23_profiles:
            p1 = s1_profiles[s1_id]
            p2 = s23_profiles[match_id]
            X.append(compute_pairwise_features(p1, p2))
            y.append(1)
            groups.append(s1_id)

    num_pos = len(X)

    # 2. Hard Negatives: Use real candidates generated by Blocking that are NOT in GT!
    print("Step 4: Mining genuine Hard Negatives from Waterfall Blocking...")
    t0 = time.time()
    
    # Fast inverted index on the ingested S2/S3 profiles
    idx_tok1 = defaultdict(list)
    idx_pin = defaultdict(list)
    idx_prefix6 = defaultdict(list)
    
    for eid, prof in s23_profiles.items():
        if prof.first_tok and len(prof.first_tok) >= 3:
            idx_tok1[(prof.country, prof.first_tok)].append(eid)
        if prof.pin:
            idx_pin[(prof.country, prof.pin)].append(eid)
        if len(prof.clean_name) >= 6:
            idx_prefix6[(prof.country, prof.clean_name[:6])].append(eid)

    neg_count = 0
    target_negs = num_pos * neg_ratio

    for s1_id in s1_target_ids:
        if s1_id not in s1_profiles:
            continue
        p1 = s1_profiles[s1_id]
        
        # Mine candidates that share keys (Hard Negatives!)
        block_cands = set()
        if p1.first_tok and len(p1.first_tok) >= 3:
            block_cands.update(idx_tok1.get((p1.country, p1.first_tok), []))
        if p1.pin:
            block_cands.update(idx_pin.get((p1.country, p1.pin), []))
        if len(p1.clean_name) >= 6:
            block_cands.update(idx_prefix6.get((p1.country, p1.clean_name[:6]), []))
            
        # Filter out true positives to get genuine hard distractors
        hard_distractors = [c for c in block_cands if (s1_id, c) not in gt_pairs and c in s23_profiles]
        
        for cand_id in hard_distractors[:neg_ratio]:
            p2 = s23_profiles[cand_id]
            X.append(compute_pairwise_features(p1, p2))
            y.append(0)
            groups.append(s1_id)
            neg_count += 1
            if neg_count >= target_negs:
                break
        if neg_count >= target_negs:
            break

    # If blocking negatives were too few, supplement with same-PIN/street negatives
    if neg_count < target_negs:
        s23_by_country = defaultdict(list)
        for eid, prof in s23_profiles.items():
            s23_by_country[prof.country].append(prof)
        for s1_id in s1_target_ids:
            p1 = s1_profiles[s1_id]
            ctry_pool = s23_by_country.get(p1.country, [])
            if not ctry_pool:
                continue
            sample = np.random.choice(ctry_pool, size=min(len(ctry_pool), 2), replace=False)
            for cand in sample:
                if (s1_id, cand.eid) not in gt_pairs:
                    X.append(compute_pairwise_features(p1, cand))
                    y.append(0)
                    groups.append(s1_id)
                    neg_count += 1
                    if neg_count >= target_negs:
                        break
            if neg_count >= target_negs:
                break

    print(f"  Matrix ready in {time.time() - t0:.2f}s: Positives={num_pos:,}, Hard Negatives={neg_count:,}, Total={len(X):,}")
    return np.array(X), np.array(y), np.array(groups)


def run_tournament(X: np.ndarray, y: np.ndarray, groups: np.ndarray, output_model_path: str = "models/final_matcher.pkl"):
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)

    print("\n" + "=" * 95)
    print("                    MODEL TOURNAMENT & ENSEMBLE BENCHMARK")
    print("=" * 95)

    # GroupKFold (80% Train, 20% Val by entity_id)
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Train set: {len(X_train):,} pairs | Validation set: {len(X_val):,} pairs (Group-split, zero leakage)")

    # Define Candidate Models
    models = {
        "1. Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
        "2. Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "3. XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss"),
        "4. CatBoost": CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05, random_seed=42, verbose=0),
        "5. LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    }

    results = []
    val_predictions = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Predict probabilities on validation fold
        t0 = time.time()
        probs = model.predict_proba(X_val)[:, 1]
        infer_time = time.time() - t0

        val_predictions[name] = probs
        fitted_models[name] = model

        # Optimize Threshold for F_0.5
        best_th, best_f05, best_p, best_r = find_best_threshold(y_val, probs)

        results.append({
            "Model": name,
            "Optimal Thresh": f"{best_th:.2f}",
            "Precision": f"{best_p * 100:.2f}%",
            "Recall": f"{best_r * 100:.2f}%",
            "F_0.5 Score": f"{best_f05:.4f}",
            "Fit Time": f"{fit_time:.2f}s",
            "Inference Time": f"{infer_time * 1000:.1f}ms"
        })

    # -------------------------------------------------------------
    # ENSEMBLE BENCHMARKS
    # -------------------------------------------------------------
    print("\nEvaluating Ensembles...")

    # Ensemble A: Top 2 (LightGBM + CatBoost)
    prob_lgb = val_predictions["5. LightGBM"]
    prob_cat = val_predictions["4. CatBoost"]
    prob_xgb = val_predictions["3. XGBoost"]

    prob_ens_top2 = 0.5 * prob_lgb + 0.5 * prob_cat
    th_ens2, f05_ens2, p_ens2, r_ens2 = find_best_threshold(y_val, prob_ens_top2)

    results.append({
        "Model": "⭐ 6. Ensemble (LightGBM + CatBoost)",
        "Optimal Thresh": f"{th_ens2:.2f}",
        "Precision": f"{p_ens2 * 100:.2f}%",
        "Recall": f"{r_ens2 * 100:.2f}%",
        "F_0.5 Score": f"{f05_ens2:.4f}",
        "Fit Time": "Combined",
        "Inference Time": "Fast (~2x)"
    })

    # Ensemble B: Top 3 (LightGBM + CatBoost + XGBoost)
    prob_ens_top3 = (prob_lgb + prob_cat + prob_xgb) / 3.0
    th_ens3, f05_ens3, p_ens3, r_ens3 = find_best_threshold(y_val, prob_ens_top3)

    results.append({
        "Model": "⭐ 7. Ensemble (LightGBM + CatBoost + XGBoost)",
        "Optimal Thresh": f"{th_ens3:.2f}",
        "Precision": f"{p_ens3 * 100:.2f}%",
        "Recall": f"{r_ens3 * 100:.2f}%",
        "F_0.5 Score": f"{f05_ens3:.4f}",
        "Fit Time": "Combined",
        "Inference Time": "Moderate (~3x)"
    })

    report_df = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print("                     FINAL MODEL BENCHMARK RESULTS (MACRO F_0.5)")
    print("=" * 95)
    print(report_df.to_string(index=False))
    print("=" * 95)

    # Save Results Table to CSV for Documentation
    report_df.to_csv("model_tournament_results.csv", index=False)
    print("Results table saved to: model_tournament_results.csv")

    # Save Winning Ensemble Pipeline
    bundle = {
        "lgbm": fitted_models["5. LightGBM"],
        "catboost": fitted_models["4. CatBoost"],
        "xgboost": fitted_models["3. XGBoost"],
        "optimal_threshold_ens2": th_ens2,
        "optimal_threshold_ens3": th_ens3,
        "optimal_threshold_lgbm": float(results[4]["Optimal Thresh"]),
        "feature_names": FEATURE_NAMES
    }
    with open(output_model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nFinal Model Bundle saved to: {output_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Entity Resolution Model Tournament")
    parser.add_argument("--gt", type=str, default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--s1", type=str, default="dataset/train/train_source1.tsv")
    parser.add_argument("--s2", type=str, default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3", type=str, default="dataset/train/train_source3.tsv")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--output", type=str, default="models/final_matcher.pkl")
    args = parser.parse_args()

    X, y, groups = prepare_group_dataset(
        gt_path=args.gt,
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        num_s1_samples=args.samples
    )
    run_tournament(X, y, groups, output_model_path=args.output)


if __name__ == "__main__":
    main()
