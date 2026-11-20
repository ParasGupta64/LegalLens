"""
prepare_multilingual_training_corpora.py
Scales the JUDGEBERT training corpus preparation pipeline for Multilingual_JUDGE.
Handles:
- Train / Dev / Test 60-10-30 stratified splits across languages and domains.
- Data Augmentation with multilingual identical pairs (Score 10.0) and unrelated pairs (Score 1.0).
- Sanity Check Holdout sets (holdout_identical_Multilingual_JUDGE.jsonl and holdout_unrelated_Multilingual_JUDGE.jsonl).
- Preserves context_text for Context-Aware vs. Sentence-Only comparative experiments.
"""

import json
import os
import random

from datasets import Dataset, DatasetDict, concatenate_datasets

SEED = 42
random.seed(SEED)


def load_multilingual_raw_dataset(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def create_stratified_splits(records: list, seed: int = 42):
    random.seed(seed)
    
    # Stratify by language and domain
    buckets = {}
    for r in records:
        key = (r["language"], r["domain"])
        buckets.setdefault(key, []).append(r)
        
    train_list, dev_list, test_list = [], [], []
    for key, group in buckets.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * 0.60)
        n_dev = int(n * 0.10)
        
        train_list.extend(group[:n_train])
        dev_list.extend(group[n_train:n_train + n_dev])
        test_list.extend(group[n_train + n_dev:])
        
    random.shuffle(train_list)
    random.shuffle(dev_list)
    random.shuffle(test_list)
    
    return train_list, dev_list, test_list


def create_identical_augmentation(records: list) -> list:
    """Creates identical pairs (Score = 10.0) for both English and Hindi."""
    identical_pairs = []
    for r in records:
        identical_pairs.append({
            "pair_id": f"IDENT_{r['pair_id']}",
            "language": r["language"],
            "domain": r["domain"],
            "context_text": r["context_text"],
            "original_text": r["original_text"],
            "simplified_text": r["original_text"],  # identical
            "original": r["original_text"],
            "simplification": r["original_text"],
            "lmp_score": 10.0,
            "label": 10.0,
            "decision": "Preserved",
            "error_types": ["none"],
            "explanation": "Identical sentence sanity pair: Legal meaning is 100% preserved."
        })
    return identical_pairs


def create_unrelated_augmentation(records: list) -> list:
    """Creates completely unrelated legal pairs (Score = 1.0) within the same language."""
    unrelated_pairs = []
    
    by_lang = {"en": [r for r in records if r["language"] == "en"],
               "hi": [r for r in records if r["language"] == "hi"]}
    
    for r in records:
        pool = by_lang[r["language"]]
        # Pick another record from a different domain or far away index
        candidates = [c for c in pool if c["domain"] != r["domain"]]
        if not candidates:
            candidates = [c for c in pool if c["pair_id"] != r["pair_id"]]
        
        chosen = random.choice(candidates)
        
        unrelated_pairs.append({
            "pair_id": f"UNREL_{r['pair_id']}",
            "language": r["language"],
            "domain": f"{r['domain']}_vs_{chosen['domain']}",
            "context_text": r["context_text"],
            "original_text": r["original_text"],
            "simplified_text": chosen["original_text"],  # completely unrelated legal clause
            "original": r["original_text"],
            "simplification": chosen["original_text"],
            "lmp_score": 1.0,
            "label": 1.0,
            "decision": "Not Preserved",
            "error_types": ["contradiction", "omission", "scope_shift"],
            "explanation": "Unrelated legal sentence sanity pair: Zero legal semantic correspondence."
        })
    return unrelated_pairs


def format_record(r: dict) -> dict:
    """Standardizes keys for HuggingFace Trainer and evaluator compatibility."""
    item = dict(r)
    item["original"] = r["original_text"]
    item["simplification"] = r["simplified_text"]
    item["label"] = float(r["lmp_score"])
    return item


def main():
    root_datastore = os.path.join("datastore")
    src_file = os.path.join(root_datastore, "released", "Multilingual_JUDGE.jsonl")
    training_dir = os.path.join(root_datastore, "training")
    os.makedirs(training_dir, exist_ok=True)
    
    print(f"Loading master dataset from {src_file}...")
    records = load_multilingual_raw_dataset(src_file)
    print(f"Loaded {len(records)} pairs.")
    
    # 1. Base Stratified Splits (without DA)
    train_records, dev_records, test_records = create_stratified_splits(records, seed=SEED)
    print(f"Splits without DA -> Train: {len(train_records)}, Dev: {len(dev_records)}, Test: {len(test_records)}")
    
    # Format records with standard fields
    train_records = [format_record(r) for r in train_records]
    dev_records = [format_record(r) for r in dev_records]
    test_records = [format_record(r) for r in test_records]
    
    # Save standard splits
    with open(os.path.join(training_dir, "train_Multilingual_JUDGE.jsonl"), "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(os.path.join(training_dir, "dev_Multilingual_JUDGE.jsonl"), "w", encoding="utf-8") as f:
        for r in dev_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(os.path.join(training_dir, "test_Multilingual_JUDGE.jsonl"), "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 2. Generate Sanity Check Holdout Sets (50 Identical, 50 Unrelated)
    print("Generating Holdout Sanity Check Sets (50 Identical, 50 Unrelated)...")
    sample_for_holdout = test_records[:50]
    holdout_identical = create_identical_augmentation(sample_for_holdout)
    holdout_unrelated = create_unrelated_augmentation(sample_for_holdout)
    
    with open(os.path.join(training_dir, "holdout_identical_Multilingual_JUDGE.jsonl"), "w", encoding="utf-8") as f:
        for r in holdout_identical:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(os.path.join(training_dir, "holdout_unrelated_Multilingual_JUDGE.jsonl"), "w", encoding="utf-8") as f:
        for r in holdout_unrelated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 3. Data Augmentation for Training Sets (DA Pipeline)
    print("Generating Data Augmentation for Train/Dev/Test sets...")
    train_da_identical = create_identical_augmentation(train_records)
    train_da_unrelated = create_unrelated_augmentation(train_records)
    
    dev_da_identical = create_identical_augmentation(dev_records)
    dev_da_unrelated = create_unrelated_augmentation(dev_records)
    
    test_da_identical = create_identical_augmentation(test_records)
    test_da_unrelated = create_unrelated_augmentation(test_records)
    
    train_with_da = train_records + train_da_identical + train_da_unrelated
    dev_with_da = dev_records + dev_da_identical + dev_da_unrelated
    test_with_da = test_records + test_da_identical + test_da_unrelated
    
    random.shuffle(train_with_da)
    random.shuffle(dev_with_da)
    random.shuffle(test_with_da)
    
    print(f"Splits WITH DA -> Train: {len(train_with_da)}, Dev: {len(dev_with_da)}, Test: {len(test_with_da)}")
    
    with open(os.path.join(training_dir, "train_Multilingual_JUDGE_with_data_augmentation.jsonl"), "w", encoding="utf-8") as f:
        for r in train_with_da:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(os.path.join(training_dir, "dev_Multilingual_JUDGE_with_data_augmentation.jsonl"), "w", encoding="utf-8") as f:
        for r in dev_with_da:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(os.path.join(training_dir, "test_Multilingual_JUDGE_with_data_augmentation.jsonl"), "w", encoding="utf-8") as f:
        for r in test_with_da:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print("\nPhase 1 Data Preparation Complete!")
    print(f"All files successfully generated in: {training_dir}")


if __name__ == "__main__":
    main()
