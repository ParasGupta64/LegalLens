"""
run_all_experiments.py
Master experimental orchestration script executing all 5 mandatory experiments
required by Dr. Surabhi Solanki's syllabus (CSET 346 - NLP):

- Experiment 1: Two Generic Semantic Similarity Baselines (Pretrained SBERT + TF-IDF Lexical) vs. JUDGEBERT
- Experiment 2: Cross-Lingual Evaluation (English vs. Hindi Pearson r & RMSE)
- Experiment 3: Context-Aware Evaluation (Sentence-Only vs. Contextual Input + Domain Breakdown)
- Experiment 4: Sanity Checks (Identical Pairs 10.0 & Unrelated Pairs 1.0)
- Experiment 5: Qualitative Error Analysis (15 Legally Difficult Case Studies via Live Inference)
"""

import argparse
import json
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from context_preprocessor import score_to_decision
from multilingual_judgebert import (
    LegalPairsDataset,
    MultilingualJUDGEBERT,
    collate_legal_batch,
    evaluate_model,
    predict_single_pair,
    save_checkpoint,
    train_multilingual_judgebert,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_jsonl(filepath: str) -> list:
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_sbert_scores(records: list, pretrained_encoder, tokenizer, device: torch.device) -> np.ndarray:
    """Computes un-finetuned pretrained SBERT cosine similarity mapped to [1.0, 10.0]."""
    pretrained_encoder.eval()
    scores = []
    with torch.no_grad():
        for r in records:
            orig = r.get("original_text") or r.get("original", "")
            simp = r.get("simplified_text") or r.get("simplification", "")

            enc_orig = tokenizer(orig, return_tensors="pt", truncation=True, max_length=256).to(device)
            enc_simp = tokenizer(simp, return_tensors="pt", truncation=True, max_length=256).to(device)

            emb_orig = mean_pooling(pretrained_encoder(**enc_orig).last_hidden_state, enc_orig["attention_mask"])
            emb_simp = mean_pooling(pretrained_encoder(**enc_simp).last_hidden_state, enc_simp["attention_mask"])

            cos_sim = F.cosine_similarity(emb_orig, emb_simp).item()
            sbert_lmp = min(10.0, max(1.0, 1.0 + 9.0 * max(0.0, cos_sim)))
            scores.append(sbert_lmp)
    return np.array(scores)


def compute_tfidf_scores(records: list) -> np.ndarray:
    """Baseline 2: Multilingual Character N-gram TF-IDF Cosine Similarity mapped to [1.0, 10.0]."""
    scores = []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    all_texts = []
    for r in records:
        all_texts.append(r.get("original_text") or r.get("original", ""))
        all_texts.append(r.get("simplified_text") or r.get("simplification", ""))
    vec.fit(all_texts)

    for r in records:
        orig = r.get("original_text") or r.get("original", "")
        simp = r.get("simplified_text") or r.get("simplification", "")
        mat = vec.transform([orig, simp]).toarray()
        denom = np.linalg.norm(mat[0]) * np.linalg.norm(mat[1])
        sim = float(np.dot(mat[0], mat[1]) / denom) if denom > 1e-9 else 0.0
        scores.append(min(10.0, max(1.0, 1.0 + 9.0 * sim)))
    return np.array(scores)


def run_experiment_1_baselines(test_records: list, pretrained_encoder, tokenizer, device: torch.device):
    """Experiment 1: Generic Pretrained SBERT & Lexical TF-IDF Baselines vs. Human LMP."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: Two Generic Semantic Similarity Baselines vs. JUDGEBERT", flush=True)
    print("=" * 70, flush=True)

    ground_truth = np.array([float(r.get("label", r.get("lmp_score", 5.0))) for r in test_records])

    # Baseline 1: Generic Pretrained SBERT (NOT fine-tuned)
    sbert_scores = compute_sbert_scores(test_records, pretrained_encoder, tokenizer, device)
    sbert_p, _ = pearsonr(ground_truth, sbert_scores)
    sbert_s, _ = spearmanr(ground_truth, sbert_scores)
    sbert_res = {
        "model": "Baseline 1: Generic Pretrained SBERT (Cosine Similarity)",
        "rmse": round(float(root_mean_squared_error(ground_truth, sbert_scores)), 4),
        "mae": round(float(mean_absolute_error(ground_truth, sbert_scores)), 4),
        "pearson_r": round(float(sbert_p), 4),
        "spearman_rho": round(float(sbert_s), 4),
        "mean_predicted_score": round(float(np.mean(sbert_scores)), 2),
        "mean_ground_truth": round(float(np.mean(ground_truth)), 2),
    }

    # Baseline 2: Lexical Character N-gram TF-IDF Cosine Similarity
    tfidf_scores = compute_tfidf_scores(test_records)
    tfidf_p, _ = pearsonr(ground_truth, tfidf_scores)
    tfidf_s, _ = spearmanr(ground_truth, tfidf_scores)
    tfidf_res = {
        "model": "Baseline 2: Lexical TF-IDF Char N-gram Similarity",
        "rmse": round(float(root_mean_squared_error(ground_truth, tfidf_scores)), 4),
        "mae": round(float(mean_absolute_error(ground_truth, tfidf_scores)), 4),
        "pearson_r": round(float(tfidf_p), 4),
        "spearman_rho": round(float(tfidf_s), 4),
        "mean_predicted_score": round(float(np.mean(tfidf_scores)), 2),
        "mean_ground_truth": round(float(np.mean(ground_truth)), 2),
    }

    print(f"Baseline 1 (Generic SBERT) -> Pearson r: {sbert_res['pearson_r']:.4f} | RMSE: {sbert_res['rmse']:.4f} | Mean Pred: {sbert_res['mean_predicted_score']}/10", flush=True)
    print(f"Baseline 2 (Lexical TFIDF) -> Pearson r: {tfidf_res['pearson_r']:.4f} | RMSE: {tfidf_res['rmse']:.4f} | Mean Pred: {tfidf_res['mean_predicted_score']}/10", flush=True)
    return sbert_res, tfidf_res, sbert_scores


def run_experiment_2_and_3(
    train_records: list,
    dev_records: list,
    test_records: list,
    device: torch.device,
    epochs: int = 6,
    freeze_bottom_layers: int = 8,
):
    """Experiment 2 (Cross-Lingual) & Experiment 3 (Sentence-Only vs Context-Aware)."""
    print("\n" + "=" * 70, flush=True)
    print("TRAINING MULTILINGUAL JUDGEBERT (Sentence-Only & Context-Aware)", flush=True)
    print("=" * 70, flush=True)

    # 1. Train Sentence-Only Model
    print("\n--- Training Model 1: Multilingual JUDGEBERT (Sentence-Only) ---", flush=True)
    model_sent, tokenizer = train_multilingual_judgebert(
        train_records,
        dev_records,
        mode="sentence_only",
        model_name=BASE_MODEL_NAME,
        epochs=epochs,
        batch_size=16,
        freeze_bottom_layers=freeze_bottom_layers,
        device=device,
    )
    save_checkpoint(model_sent, tokenizer, os.path.join("checkpoints", "sentence_only"), mode="sentence_only")

    # 2. Train Context-Aware Model
    print("\n--- Training Model 2: Multilingual JUDGEBERT (Context-Aware) ---", flush=True)
    model_ctx, _ = train_multilingual_judgebert(
        train_records,
        dev_records,
        mode="context_aware",
        model_name=BASE_MODEL_NAME,
        epochs=epochs,
        batch_size=16,
        freeze_bottom_layers=freeze_bottom_layers,
        device=device,
    )
    save_checkpoint(model_ctx, tokenizer, os.path.join("checkpoints", "context_aware"), mode="context_aware")

    # Evaluate on Test Sets
    test_ds_sent = LegalPairsDataset(test_records, tokenizer, mode="sentence_only")
    test_loader_sent = DataLoader(test_ds_sent, batch_size=16, shuffle=False, collate_fn=lambda b: collate_legal_batch(b, tokenizer))
    metrics_sent = evaluate_model(model_sent, test_loader_sent, device)

    test_ds_ctx = LegalPairsDataset(test_records, tokenizer, mode="context_aware")
    test_loader_ctx = DataLoader(test_ds_ctx, batch_size=16, shuffle=False, collate_fn=lambda b: collate_legal_batch(b, tokenizer))
    metrics_ctx = evaluate_model(model_ctx, test_loader_ctx, device)

    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3 RESULTS: Sentence-Only vs. Context-Aware Evaluation", flush=True)
    print("=" * 70, flush=True)
    print(f"Sentence-Only JUDGEBERT -> Pearson r: {metrics_sent['pearson_r']:.4f} | RMSE: {metrics_sent['rmse']:.4f} | MAE: {metrics_sent['mae']:.4f} | Decision Acc: {metrics_sent['decision_acc']*100:.1f}%", flush=True)
    print(f"Context-Aware JUDGEBERT -> Pearson r: {metrics_ctx['pearson_r']:.4f} | RMSE: {metrics_ctx['rmse']:.4f} | MAE: {metrics_ctx['mae']:.4f} | Decision Acc: {metrics_ctx['decision_acc']*100:.1f}%", flush=True)

    # Experiment 2: Cross-Lingual Breakdown (English, Hindi and Punjabi)
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2 RESULTS: Cross-Lingual Evaluation (English, Hindi & Punjabi)", flush=True)
    print("=" * 70, flush=True)

    language_metrics = {}
    language_codes = {"english": "en", "hindi": "hi", "punjabi": "pa"}
    for language_name, language_code in language_codes.items():
        lang_records = [r for r in test_records if str(r.get("language", "")).lower() in (language_code, language_name)]
        if len(lang_records) < 2:
            language_metrics[language_name] = {
                "status": "not_evaluated",
                "count": len(lang_records),
                "reason": "At least two labelled test records are required."
            }
            print(f"{language_name.title()} Sub-Corpus (N={len(lang_records)}) -> NOT EVALUATED: insufficient labelled test records.", flush=True)
            continue

        lang_ds = LegalPairsDataset(lang_records, tokenizer, mode="context_aware")
        lang_loader = DataLoader(lang_ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate_legal_batch(b, tokenizer))
        metrics = evaluate_model(model_ctx, lang_loader, device)
        language_metrics[language_name] = {
            "status": "evaluated",
            "count": len(lang_records),
            "pearson_r": round(float(metrics["pearson_r"]), 4),
            "rmse": round(float(metrics["rmse"]), 4),
            "mae": round(float(metrics["mae"]), 4),
            "acc": round(float(metrics["decision_acc"]), 4),
        }
        print(f"{language_name.title()} Sub-Corpus (N={len(lang_records)}) -> Pearson r: {metrics['pearson_r']:.4f} | RMSE: {metrics['rmse']:.4f} | Acc: {metrics['decision_acc']*100:.1f}%", flush=True)

    en_metrics = language_metrics["english"]
    hi_metrics = language_metrics["hindi"]
    pa_metrics = language_metrics["punjabi"]

    domain_results = {}
    for dom in ["contract", "statute", "judgment"]:
        dom_records = [r for r in test_records if r["domain"] == dom]
        if dom_records:
            dom_ds = LegalPairsDataset(dom_records, tokenizer, mode="context_aware")
            dom_loader = DataLoader(dom_ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate_legal_batch(b, tokenizer))
            m = evaluate_model(model_ctx, dom_loader, device)
            domain_results[dom] = {
                "count": len(dom_records),
                "pearson_r": round(float(m["pearson_r"]), 4),
                "rmse": round(float(m["rmse"]), 4),
                "acc": round(float(m["decision_acc"]), 4),
            }
            print(f"Domain: {dom.capitalize():<10} (N={len(dom_records)}) -> Pearson r: {m['pearson_r']:.4f} | RMSE: {m['rmse']:.4f} | Acc: {m['decision_acc']*100:.1f}%", flush=True)

    return {
        "model_sent": model_sent,
        "model_ctx": model_ctx,
        "tokenizer": tokenizer,
        "metrics_sent": metrics_sent,
        "metrics_ctx": metrics_ctx,
        "en_metrics": en_metrics,
        "hi_metrics": hi_metrics,
        "domain_results": domain_results,
    }


def run_experiment_4_sanity_checks(
    model: MultilingualJUDGEBERT,
    pretrained_encoder,
    tokenizer,
    holdout_ident_file: str,
    holdout_unrel_file: str,
    device: torch.device,
):
    """Experiment 4: Sanity Checks (Identical Pairs vs Unrelated Pairs)."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 4: Sanity Checks on Identical & Unrelated Pairs", flush=True)
    print("=" * 70, flush=True)

    ident_records = load_jsonl(holdout_ident_file)
    unrel_records = load_jsonl(holdout_unrel_file)

    ident_ds = LegalPairsDataset(ident_records, tokenizer, mode="context_aware")
    ident_loader = DataLoader(ident_ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate_legal_batch(b, tokenizer))
    ident_metrics = evaluate_model(model, ident_loader, device)

    unrel_ds = LegalPairsDataset(unrel_records, tokenizer, mode="context_aware")
    unrel_loader = DataLoader(unrel_ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate_legal_batch(b, tokenizer))
    unrel_metrics = evaluate_model(model, unrel_loader, device)

    sbert_ident = compute_sbert_scores(ident_records, pretrained_encoder, tokenizer, device)
    sbert_unrel = compute_sbert_scores(unrel_records, pretrained_encoder, tokenizer, device)

    ident_preds = ident_metrics["predictions"]
    unrel_preds = unrel_metrics["predictions"]

    ident_pass_rate = float(np.mean(ident_preds >= 8.5)) * 100.0
    unrel_pass_rate = float(np.mean(unrel_preds <= 2.5)) * 100.0

    print(f"Generic SBERT Baseline -> Identical Mean: {np.mean(sbert_ident):.2f}/10 | Unrelated Mean: {np.mean(sbert_unrel):.2f}/10", flush=True)
    print(f"JUDGEBERT Identical (Target: 10.0) -> Mean Pred: {np.mean(ident_preds):.2f}/10 | Pass Rate (>= 8.5): {ident_pass_rate:.1f}%", flush=True)
    print(f"JUDGEBERT Unrelated (Target: 1.0)  -> Mean Pred: {np.mean(unrel_preds):.2f}/10 | Pass Rate (<= 2.5): {unrel_pass_rate:.1f}%", flush=True)

    return {
        "generic_sbert_identical_mean": round(float(np.mean(sbert_ident)), 2),
        "generic_sbert_unrelated_mean": round(float(np.mean(sbert_unrel)), 2),
        "identical_mean_pred": round(float(np.mean(ident_preds)), 2),
        "identical_rmse": round(float(ident_metrics["rmse"]), 4),
        "identical_pass_rate": round(ident_pass_rate, 2),
        "unrelated_mean_pred": round(float(np.mean(unrel_preds)), 2),
        "unrelated_rmse": round(float(unrel_metrics["rmse"]), 4),
        "unrelated_pass_rate": round(unrel_pass_rate, 2),
    }


def run_experiment_5_case_studies(
    model: MultilingualJUDGEBERT,
    pretrained_encoder,
    tokenizer,
    records: list,
    device: torch.device,
):
    """Experiment 5: Qualitative Error Analysis for 15 Legally Difficult Case Studies."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 5: Qualitative Error Analysis (15 Case Studies)", flush=True)
    print("=" * 70, flush=True)

    # Select case studies from English, Hindi and Punjabi where records exist.
    selected = []
    for lang_code in ("en", "hi", "pa"):
        lang_records = [r for r in records if str(r.get("language", "")).lower() == lang_code]
        selected.extend(lang_records[:5])
    if len(selected) < 15:
        selected_ids = {r.get("pair_id") for r in selected}
        selected.extend([r for r in records if r.get("pair_id") not in selected_ids][:15-len(selected)])
    selected = selected[:15]

    sbert_case_scores = compute_sbert_scores(selected, pretrained_encoder, tokenizer, device)
    case_studies = []

    for i, (r, sbert_sc) in enumerate(zip(selected, sbert_case_scores), 1):
        pred = predict_single_pair(
            model,
            tokenizer,
            original=r["original_text"],
            simplification=r["simplified_text"],
            context=r["context_text"],
            lang=r["language"],
            mode="context_aware",
            device=device,
        )
        gt_score = float(r.get("label", r.get("lmp_score", 5.0)))

        case = {
            "case_number": i,
            "pair_id": r["pair_id"],
            "language": r["language"],
            "domain": r["domain"],
            "context": r["context_text"],
            "original_clause": r["original_text"],
            "simplified_clause": r["simplified_text"],
            "generic_sbert_score": round(float(sbert_sc), 2),
            "human_ground_truth": {
                "lmp_score": gt_score,
                "decision": r["decision"],
                "error_types": r["error_types"],
                "human_explanation": r["explanation"],
            },
            "judgebert_prediction": {
                "predicted_lmp_score": pred["lmp_score"],
                "predicted_decision": pred["decision"],
                "diagnosed_error_types": pred["error_types"],
                "automated_explanation": pred["explanation"],
            },
        }
        case_studies.append(case)

        print(f"\n[Case {i:02d} | {r['pair_id']} | {r['language'].upper()} | {r['domain'].capitalize()}]", flush=True)
        print(f"Original:     {r['original_text'][:85]}...", flush=True)
        print(f"Simplified:   {r['simplified_text'][:85]}...", flush=True)
        print(f"Ground Truth: {gt_score}/10 ({r['decision']}) | Errors: {r['error_types']}", flush=True)
        print(f"SBERT Score:  {sbert_sc:.1f}/10 | JUDGEBERT: {pred['lmp_score']:.1f}/10 ({pred['decision']}) | Diagnosed: {pred['error_types']}", flush=True)
        print(f"Explanation:  {pred['explanation']}", flush=True)

    return case_studies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs per model.")
    parser.add_argument("--freeze_layers", type=int, default=8, help="Bottom encoder layers to freeze.")
    args = parser.parse_args()

    root_datastore = os.path.join("datastore", "training")
    out_dir = os.path.join("predictions")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing experiments pipeline on device: {device}", flush=True)

    train_file = os.path.join(root_datastore, "train_Multilingual_JUDGE_with_data_augmentation.jsonl")
    dev_file = os.path.join(root_datastore, "dev_Multilingual_JUDGE_with_data_augmentation.jsonl")
    test_file = os.path.join(root_datastore, "test_Multilingual_JUDGE.jsonl")
    holdout_ident_file = os.path.join(root_datastore, "holdout_identical_Multilingual_JUDGE.jsonl")
    holdout_unrel_file = os.path.join(root_datastore, "holdout_unrelated_Multilingual_JUDGE.jsonl")

    train_records = load_jsonl(train_file)
    dev_records = load_jsonl(dev_file)
    test_records = load_jsonl(test_file)

    # Load un-finetuned pretrained encoder for true Baseline 1 comparison
    print(f"Loading un-finetuned pretrained encoder for Baseline 1 ({BASE_MODEL_NAME})...", flush=True)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    pretrained_encoder = AutoModel.from_pretrained(BASE_MODEL_NAME).to(device)

    # 1. Experiment 1: Generic Baselines
    sbert_res, tfidf_res, _ = run_experiment_1_baselines(test_records, pretrained_encoder, pretrained_tokenizer, device)

    # 2. Experiments 2 & 3: Train & Evaluate Multilingual JUDGEBERT Models
    m_res = run_experiment_2_and_3(
        train_records, dev_records, test_records, device,
        epochs=args.epochs, freeze_bottom_layers=args.freeze_layers
    )
    model_ctx = m_res["model_ctx"]
    tokenizer = m_res["tokenizer"]

    # 3. Experiment 4: Sanity Checks
    exp4_res = run_experiment_4_sanity_checks(
        model_ctx, pretrained_encoder, tokenizer, holdout_ident_file, holdout_unrel_file, device
    )

    # 4. Experiment 5: Qualitative Error Analysis (15 Case Studies)
    master_records = load_jsonl(os.path.join("datastore", "released", "Multilingual_JUDGE.jsonl"))
    exp5_cases = run_experiment_5_case_studies(model_ctx, pretrained_encoder, tokenizer, master_records, device)

    summary_report = {
        "project_title": "LegalLens: Context-Aware Legal Meaning Preservation",
        "course": "CSET 346 – Natural Language Processing",
        "instructor": "Dr. Surabhi Solanki",
        "experiment_1_baseline_vs_judgebert": {
            "generic_sbert_baseline": sbert_res,
            "lexical_tfidf_baseline": tfidf_res,
            "multilingual_judgebert": {
                "rmse": round(float(m_res["metrics_ctx"]["rmse"]), 4),
                "mae": round(float(m_res["metrics_ctx"]["mae"]), 4),
                "pearson_r": round(float(m_res["metrics_ctx"]["pearson_r"]), 4),
                "spearman_rho": round(float(m_res["metrics_ctx"]["spearman_rho"]), 4),
                "decision_acc": round(float(m_res["metrics_ctx"]["decision_acc"]), 4),
                "decision_macro_f1": round(float(m_res["metrics_ctx"]["decision_macro_f1"]), 4),
            },
        },
        "experiment_2_cross_lingual": m_res["language_metrics"],
        "experiment_3_context_aware_generalization": {
            "sentence_only": {
                "rmse": round(float(m_res["metrics_sent"]["rmse"]), 4),
                "mae": round(float(m_res["metrics_sent"]["mae"]), 4),
                "pearson_r": round(float(m_res["metrics_sent"]["pearson_r"]), 4),
                "acc": round(float(m_res["metrics_sent"]["decision_acc"]), 4),
            },
            "context_aware": {
                "rmse": round(float(m_res["metrics_ctx"]["rmse"]), 4),
                "mae": round(float(m_res["metrics_ctx"]["mae"]), 4),
                "pearson_r": round(float(m_res["metrics_ctx"]["pearson_r"]), 4),
                "acc": round(float(m_res["metrics_ctx"]["decision_acc"]), 4),
            },
            "domain_breakdown": m_res["domain_results"],
        },
        "experiment_4_sanity_checks": exp4_res,
        "experiment_5_case_studies_count": len(exp5_cases),
    }

    with open(os.path.join(out_dir, "experiment_results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "experiment_5_error_case_studies.json"), "w", encoding="utf-8") as f:
        json.dump(exp5_cases, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70, flush=True)
    print("ALL 5 EXPERIMENTS COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Checkpoints saved to: checkpoints/sentence_only and checkpoints/context_aware", flush=True)
    print(f"Results saved to: {out_dir}/experiment_results_summary.json", flush=True)
    print(f"Case studies saved to: {out_dir}/experiment_5_error_case_studies.json", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
