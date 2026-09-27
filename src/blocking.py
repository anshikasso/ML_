"""
High-Performance Production Waterfall Blocking for Entity Resolution.

Optimizations:
  1. Direct single-pass stream indexing (no giant intermediary lists in RAM)
  2. Inverted index frequency filtering (caps high-frequency words to prevent candidate bloat)
  3. Country-partitioned character 3-gram TF-IDF fallback for orphan records
  4. Supports fast development sampling (--sample-s1, --sample-s23)
  5. Strictly complies with candidate_pairs.tsv formatting rules
"""

import os
import re
import sys
import time
import argparse
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import unicodedata

try:
    from src.features import transliterate_indic, has_indic_characters, normalize_digits
except ImportError:
    from features import transliterate_indic, has_indic_characters, normalize_digits

# Legal noise suffixes across US, India, and France
LEGAL_SUFFIXES = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated", "llc", "llp",
    "corp", "corporation", "co", "company", "sa", "sarl", "sas", "enterprises",
    "services", "solutions", "group", "holdings", "the", "and", "center"
}

# Frequent generic business terms to avoid indexing as lone first tokens
GENERIC_STOPWORDS = {
    "hotel", "restaurant", "store", "market", "shop", "cafe", "hospital",
    "school", "college", "bank", "clinic", "pharmacy", "traders", "agency"
}


def clean_name(name: str) -> str:
    """Normalize business name: digit normalize, transliterate Indic, strip accents, remove legal noise words."""
    if not isinstance(name, str):
        return ""
    norm = normalize_digits(name)
    if has_indic_characters(norm):
        norm = transliterate_indic(norm)
    else:
        norm = unicodedata.normalize('NFKD', norm).encode('ascii', 'ignore').decode('utf-8')
    norm = norm.lower()
    norm = re.sub(r"[^\w\s]", " ", norm)
    tokens = [tok for tok in norm.split() if tok not in LEGAL_SUFFIXES]
    return " ".join(tokens) if tokens else norm.strip()


def extract_postal_code(address: str) -> str:
    """Extract 5-digit US/French zip or 6-digit Indian PIN code."""
    if not isinstance(address, str):
        return ""
    norm = normalize_digits(address)
    match = re.search(r"\b\d{5,6}\b", norm)
    return match.group(0) if match else ""


class WaterfallBlocker:
    def __init__(self, min_threshold: int = 3, max_cap: int = 15, max_bucket_size: int = 80):
        self.min_threshold = min_threshold
        self.max_cap = max_cap
        self.max_bucket_size = max_bucket_size
        
        # Tier 1 Inverted Indexes
        self.idx_tok1 = defaultdict(list)
        self.idx_pin = defaultdict(list)
        self.idx_pref5 = defaultdict(list)
        self.idx_tok2 = defaultdict(list)
        
        # Tier 2 TF-IDF Fallback structures (country -> components)
        self.fallback_eids = defaultdict(list)
        self.fallback_corpus = defaultdict(list)
        self.tfidf_vectorizers = {}
        self.tfidf_matrices = {}

    def index_s23(self, s2_path: str, s3_path: str, sample_s23: int = None):
        """Single-pass streaming indexing of S2 and S3."""
        print(f"Indexing Source 2 and Source 3...")
        t0 = time.time()
        count = 0
        max_fallback_per_country = sample_s23 if sample_s23 is not None else 25000

        for path in [s2_path, s3_path]:
            if not os.path.exists(path):
                print(f"Warning: File {path} not found.")
                continue

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline()  # Skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 4:
                        continue
                    eid, b_name, b_addr, country = parts[0], parts[1], parts[2], parts[3]
                    
                    c_name = clean_name(b_name)
                    pin = extract_postal_code(b_addr)
                    tokens = c_name.split()
                    first_tok = tokens[0] if tokens else ""

                    # Index Tier 1 keys (with bucket size safety cap)
                    if first_tok and len(first_tok) >= 3 and first_tok not in GENERIC_STOPWORDS:
                        bucket = self.idx_tok1[(country, first_tok)]
                        if len(bucket) < self.max_bucket_size:
                            bucket.append(eid)

                    if pin:
                        bucket = self.idx_pin[(country, pin)]
                        if len(bucket) < self.max_bucket_size:
                            bucket.append(eid)

                    no_space_name = "".join(tokens)
                    if len(no_space_name) >= 5:
                        bucket = self.idx_pref5[(country, no_space_name[:5])]
                        if len(bucket) < self.max_bucket_size:
                            bucket.append(eid)

                    if len(tokens) > 1 and len(tokens[1]) >= 4 and tokens[1] not in GENERIC_STOPWORDS:
                        bucket = self.idx_tok2[(country, tokens[1])]
                        if len(bucket) < self.max_bucket_size:
                            bucket.append(eid)

                    # Store for TF-IDF fallback (safely capped per country to prevent memory exhaustion)
                    if len(self.fallback_eids[country]) < max_fallback_per_country:
                        self.fallback_eids[country].append(eid)
                        self.fallback_corpus[country].append(c_name)

                    count += 1
                    if sample_s23 and count >= sample_s23:
                        break
            if sample_s23 and count >= sample_s23:
                break

        print(f"  Indexed {count:,} S2/S3 records in {time.time() - t0:.2f}s:")
        print(f"    - First Token Keys: {len(self.idx_tok1):,}")
        print(f"    - Postal Code Keys: {len(self.idx_pin):,}")
        print(f"    - Prefix-5 Keys:    {len(self.idx_pref5):,}")
        print(f"    - Second Token Keys:{len(self.idx_tok2):,}")

    def fit_tfidf_fallback(self):
        """Fit character 3-gram TF-IDF on country partitions."""
        print("Fitting country-partitioned TF-IDF Fallback models...")
        t0 = time.time()
        for country, corpus in self.fallback_corpus.items():
            if not corpus:
                continue
            # Character n-grams handle spelling errors and script variations
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_features=25000)
            X = vec.fit_transform(corpus)
            self.tfidf_vectorizers[country] = vec
            self.tfidf_matrices[country] = X
            self.fallback_eids[country] = np.array(self.fallback_eids[country])
            print(f"  [{country}] Vocabulary: {len(vec.vocabulary_):,}, Matrix shape: {X.shape}")
        print(f"TF-IDF models initialized in {time.time() - t0:.2f}s")

    def query_entity(self, eid: str, b_name: str, b_addr: str, country: str) -> list:
        """Execute Waterfall query for a single Source 1 record."""
        c_name = clean_name(b_name)
        pin = extract_postal_code(b_addr)
        tokens = c_name.split()
        first_tok = tokens[0] if tokens else ""

        # --- TIER 1: Inverted Index Hash-Join ---
        cands = set()
        if first_tok and len(first_tok) >= 3 and first_tok not in GENERIC_STOPWORDS:
            cands.update(self.idx_tok1.get((country, first_tok), []))
        if pin:
            cands.update(self.idx_pin.get((country, pin), []))
        no_space_name = "".join(tokens)
        if len(no_space_name) >= 5:
            cands.update(self.idx_pref5.get((country, no_space_name[:5]), []))
        if len(tokens) > 1 and len(tokens[1]) >= 4 and tokens[1] not in GENERIC_STOPWORDS:
            cands.update(self.idx_tok2.get((country, tokens[1]), []))

        # --- TIER 2: TF-IDF Fallback (if candidates < threshold) ---
        if len(cands) < self.min_threshold and country in self.tfidf_vectorizers and c_name:
            vec = self.tfidf_vectorizers[country]
            matrix = self.tfidf_matrices[country]
            eids = self.fallback_eids[country]

            q_vec = vec.transform([c_name])
            sims = q_vec.dot(matrix.T).toarray().ravel()
            
            if len(sims) > 0:
                k = min(self.max_cap, len(sims))
                top_idx = np.argpartition(sims, -k)[-k:]
                valid = [idx for idx in top_idx if sims[idx] > 0.15]
                cands.update(eids[valid])

        # Cap candidates to preserve efficiency ranking
        return list(cands)[:self.max_cap]


def generate_candidate_file(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str,
    min_threshold: int = 3,
    max_cap: int = 15,
    sample_s1: int = None,
    sample_s23: int = None
):
    start_time = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=" * 80)
    print("      PRODUCTION WATERFALL CANDIDATE GENERATION (BLOCKING)")
    print("=" * 80)
    print(f"S1 Path:     {s1_path}")
    print(f"S2 Path:     {s2_path}")
    print(f"S3 Path:     {s3_path}")
    print(f"Output File: {output_path}")
    print(f"Min Threshold: {min_threshold} | Max Cap: {max_cap}")
    if sample_s1 or sample_s23:
        print(f"Sample Mode: S1={sample_s1}, S2/S3={sample_s23}")
    print("-" * 80)

    # 1. Initialize and Index S2/S3
    blocker = WaterfallBlocker(min_threshold=min_threshold, max_cap=max_cap)
    blocker.index_s23(s2_path, s3_path, sample_s23=sample_s23)
    blocker.fit_tfidf_fallback()

    # 2. Process S1 and Stream to Output File
    print(f"\nProcessing Source 1 entities and writing to {output_path}...")
    t0 = time.time()
    total_s1 = 0
    total_candidates = 0
    zero_cands = 0

    with open(s1_path, "r", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        in_f.readline()  # Skip header

        for line in in_f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[0]
            b_name = parts[1] if len(parts) > 1 else ""
            b_addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""

            total_s1 += 1
            if sample_s1 and total_s1 > sample_s1:
                break

            cand_list = blocker.query_entity(eid, b_name, b_addr, country)
            c_count = len(cand_list)
            total_candidates += c_count

            if c_count == 0:
                zero_cands += 1
                out_f.write(f"{eid}\t\n")
            else:
                out_f.write(f"{eid}\t{','.join(cand_list)}\n")

            if total_s1 % 50000 == 0:
                print(f"  Processed {total_s1:,} S1 entities... (Avg cands: {total_candidates / total_s1:.1f})")

    elapsed = time.time() - t0
    avg_cands = total_candidates / total_s1 if total_s1 > 0 else 0

    print("\n" + "=" * 80)
    print("                     FINAL BLOCKING REPORT")
    print("=" * 80)
    print(f"Total S1 Records Processed:   {total_s1:,}")
    print(f"Total Candidate Pairs:        {total_candidates:,}")
    print(f"Average Candidates per S1:    {avg_cands:.2f} (Target: < 15)")
    print(f"Entities with 0 Candidates:   {zero_cands:,} ({zero_cands / total_s1 * 100:.1f}%)")
    print(f"Total Pipeline Runtime:       {time.time() - start_time:.2f}s")
    print(f"Output File:                  {output_path}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Waterfall Blocker for Entity Resolution")
    parser.add_argument("--s1", type=str, default="dataset/test/test_source1.tsv", help="Path to Source 1 TSV")
    parser.add_argument("--s2", type=str, default="dataset/test/test_source2.tsv", help="Path to Source 2 TSV")
    parser.add_argument("--s3", type=str, default="dataset/test/test_source3.tsv", help="Path to Source 3 TSV")
    parser.add_argument("--output", type=str, default="output/candidate_pairs.tsv", help="Output candidate TSV path")
    parser.add_argument("--min-threshold", type=int, default=3, help="Min candidates before TF-IDF fallback")
    parser.add_argument("--max-cap", type=int, default=15, help="Max candidates per S1 entity")
    parser.add_argument("--sample-s1", type=int, default=None, help="Sample N records from S1")
    parser.add_argument("--sample-s23", type=int, default=None, help="Sample N records from S2/S3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_candidate_file(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.output,
        min_threshold=args.min_threshold,
        max_cap=args.max_cap,
        sample_s1=args.sample_s1,
        sample_s23=args.sample_s23
    )
