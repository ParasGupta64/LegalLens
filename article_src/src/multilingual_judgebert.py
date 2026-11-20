"""
multilingual_judgebert.py
Core model architecture, training, and explainable inference engine for
"Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT".

Features:
- Multilingual transformer backbone (XLM-RoBERTa / paraphrase-multilingual-MiniLM-L12-v2).
- Multi-Task Head:
    1. Continuous LMP Regression Head (predicts 1.0 - 10.0)
    2. Preservation Decision Classifier (Preserved / Partially Preserved / Not Preserved)
    3. Diagnostic Error Classifier (9 legal error dimensions)
    4. Explainability Generator (rules-based diagnostic summary)
- Supports both Sentence-Only and Context-Aware inputs.
"""

import math
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from context_preprocessor import (
    DECISION_TO_ID,
    ID_TO_DECISION,
    ERROR_TAXONOMY,
    ERROR_TO_ID,
    ID_TO_ERROR,
    format_input_text,
    score_to_decision
)


class MultilingualJUDGEBERT(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dropout: float = 0.1,
        freeze_bottom_layers: int = 8,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_bottom_layers = freeze_bottom_layers
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Freeze embeddings and bottom transformer layers for fast, stable CPU/GPU fine-tuning
        if freeze_bottom_layers > 0:
            if hasattr(self.encoder, "embeddings"):
                for param in self.encoder.embeddings.parameters():
                    param.requires_grad = False
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                for layer in self.encoder.encoder.layer[:freeze_bottom_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # 1. Primary Continuous Regression Head (linear output, matches original JUDGEBERT)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        # Initialize bias around midpoint 5.5 for faster convergence
        nn.init.constant_(self.regression_head[-1].bias, 5.5)

        # 2. Decision Classification Head (3 classes)
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3),
        )

        # 3. Diagnostic Error Classifier (9 binary error dimensions)
        self.error_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, len(ERROR_TAXONOMY)),
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        lmp_scores = self.regression_head(pooled).squeeze(-1)
        if not self.training:
            lmp_scores = torch.clamp(lmp_scores, 1.0, 10.0)

        decision_logits = self.decision_head(pooled)
        error_logits = self.error_head(pooled)

        return {
            "lmp_scores": lmp_scores,
            "decision_logits": decision_logits,
            "error_logits": error_logits,
            "embeddings": pooled,
        }


class LegalPairsDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, mode: str = "sentence_only", max_length: int = 384):
        self.records = records
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        text_a, text_b = format_input_text(item, mode=self.mode)

        encoding = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_token_type_ids=False,
        )

        raw_score = item.get("label")
        if raw_score is None:
            raw_score = item.get("lmp_score", 5.0)
        score = float(raw_score)
        dec = item.get("decision", score_to_decision(score))
        dec_id = DECISION_TO_ID.get(dec, 1)

        err_vec = [0.0] * len(ERROR_TAXONOMY)
        for err in item.get("error_types", []):
            if err in ERROR_TO_ID:
                err_vec[ERROR_TO_ID[err]] = 1.0

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "lmp_score": score,
            "decision_id": dec_id,
            "error_vec": err_vec,
            "meta": item,
        }


def collate_legal_batch(batch, tokenizer):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]

    padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    scores = torch.tensor([x["lmp_score"] for x in batch], dtype=torch.float)
    decisions = torch.tensor([x["decision_id"] for x in batch], dtype=torch.long)
    errors = torch.tensor([x["error_vec"] for x in batch], dtype=torch.float)

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
        "lmp_score": scores,
        "decision_id": decisions,
        "error_vec": errors,
        "metas": [x["meta"] for x in batch],
    }


def generate_explanation(score: float, decision: str, detected_errors: List[str], lang: str = "en") -> str:
    """Produces explainable legal rationale based on model diagnostics."""
    if decision == "Preserved" and (not detected_errors or detected_errors == ["none"]):
        return (
            "The simplified clause preserves all core legal rights, conditions, and obligations "
            "with zero material distortion of legal consequences."
            if lang == "en" else
            "सरलीकृत खंड मूल कानूनी अधिकारों, शर्तों और दायित्वों को पूरी तरह सुरक्षित रखता है।"
        )

    parts = []
    for err in detected_errors:
        if err == "obligation_shift":
            parts.append("an altered legal duty or discretionary permission (e.g. shall vs may)")
        elif err == "condition_shift":
            parts.append("an omitted or modified prerequisite condition / trigger")
        elif err == "exception_shift":
            parts.append("an altered or omitted statutory / contractual exception")
        elif err == "dates_numbers_shift":
            parts.append("a shift in statutory deadlines, percentages, or financial amounts")
        elif err == "scope_shift":
            parts.append("an expansion or restriction of legal applicability / covered parties")
        elif err == "entity_shift":
            parts.append("a confusion or reversal of legal roles (e.g. licensor vs licensee)")
        elif err == "contradiction":
            parts.append("a direct contradiction of legal meaning or entitlement")
        elif err == "omission":
            parts.append("an omission of material legal safeguards or limiting terms")
        elif err == "hallucination":
            parts.append("unsupported legal claims introduced in the simplification")

    joined_errors = ", ".join(parts) if parts else "a shift in legal nuance"
    return f"Legal meaning is {decision.lower()} due to {joined_errors} (Predicted LMP: {score:.1f}/10)."


def evaluate_model(model: MultilingualJUDGEBERT, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []
    all_dec_preds = []
    all_dec_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids, attention_mask)
            preds = out["lmp_scores"].cpu().numpy()
            targets = batch["lmp_score"].numpy()

            dec_preds = torch.argmax(out["decision_logits"], dim=-1).cpu().numpy()
            dec_targets = batch["decision_id"].numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            all_dec_preds.extend(dec_preds.tolist())
            all_dec_targets.extend(dec_targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    rmse = float(root_mean_squared_error(all_targets, all_preds))
    mae = float(mean_absolute_error(all_targets, all_preds))

    if np.std(all_preds) > 1e-6 and np.std(all_targets) > 1e-6:
        p_corr, _ = pearsonr(all_targets, all_preds)
        s_corr, _ = spearmanr(all_targets, all_preds)
    else:
        p_corr, s_corr = 0.0, 0.0

    dec_acc = float(np.mean(np.array(all_dec_preds) == np.array(all_dec_targets)))
    macro_f1 = float(f1_score(all_dec_targets, all_dec_preds, average="macro", zero_division=0))

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": float(p_corr),
        "spearman_rho": float(s_corr),
        "decision_acc": dec_acc,
        "decision_macro_f1": macro_f1,
        "predictions": all_preds,
        "targets": all_targets,
    }


def predict_single_pair(
    model: MultilingualJUDGEBERT,
    tokenizer,
    original: str,
    simplification: str,
    context: str = "",
    lang: str = "en",
    mode: str = "context_aware",
    device: Optional[torch.device] = None,
    error_threshold: float = 0.45,
) -> Dict:
    """Runs live inference on a single (context, original, simplified) legal input."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    sample = {
        "original_text": original,
        "simplified_text": simplification,
        "context_text": context,
    }
    text_a, text_b = format_input_text(sample, mode=mode)
    enc = tokenizer(text_a, text_b, return_tensors="pt", truncation=True, max_length=384).to(device)

    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
        score = float(out["lmp_scores"].item())
        dec_id = int(torch.argmax(out["decision_logits"], dim=-1).item())
        dec_head_label = ID_TO_DECISION.get(dec_id, score_to_decision(score))
        # Align decision consistently with calibrated score bracket when regression & head agree closely
        score_bracket = score_to_decision(score)
        decision = score_bracket if abs(score - 5.0) > 0.5 and abs(score - 8.0) > 0.5 else dec_head_label

        err_probs = torch.sigmoid(out["error_logits"]).squeeze(0).cpu().numpy()
        detected = [
            (ERROR_TAXONOMY[idx], float(p))
            for idx, p in enumerate(err_probs)
            if p >= error_threshold
        ]
        detected.sort(key=lambda x: x[1], reverse=True)
        detected_errors = [name for name, _ in detected] if detected else (
            ["none"] if decision == "Preserved" else [ERROR_TAXONOMY[int(np.argmax(err_probs))]]
        )

    explanation = generate_explanation(score, decision, detected_errors, lang=lang)
    return {
        "lmp_score": round(score, 2),
        "decision": decision,
        "error_types": detected_errors,
        "error_probabilities": {ERROR_TAXONOMY[i]: round(float(p), 3) for i, p in enumerate(err_probs)},
        "explanation": explanation,
        "evaluation_mode": mode,
        "context_used": bool(mode == "context_aware" and context.strip()),
    }


def save_checkpoint(model: MultilingualJUDGEBERT, tokenizer, save_dir: str, mode: str):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model.model_name,
            "freeze_bottom_layers": model.freeze_bottom_layers,
            "mode": mode,
        },
        ckpt_path,
    )
    tokenizer.save_pretrained(save_dir)
    print(f"Saved model checkpoint to {ckpt_path}")


def load_checkpoint(save_dir: str, device: Optional[torch.device] = None) -> Tuple[MultilingualJUDGEBERT, AutoTokenizer, str]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(save_dir, "model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = MultilingualJUDGEBERT(
        model_name=ckpt["model_name"],
        freeze_bottom_layers=ckpt.get("freeze_bottom_layers", 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, ckpt.get("mode", "context_aware")


def train_multilingual_judgebert(
    train_records: List[Dict],
    dev_records: List[Dict],
    mode: str = "sentence_only",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    epochs: int = 6,
    batch_size: int = 16,
    lr_backbone: float = 5e-5,
    lr_head: float = 1e-3,
    freeze_bottom_layers: int = 8,
    device: Optional[torch.device] = None,
) -> Tuple[MultilingualJUDGEBERT, AutoTokenizer]:
    """Fine-tunes the Multilingual JUDGEBERT model using differential learning rates."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nInitializing model {model_name} on {device} (Mode: {mode}, Frozen bottom layers: {freeze_bottom_layers})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultilingualJUDGEBERT(
        model_name=model_name, freeze_bottom_layers=freeze_bottom_layers
    ).to(device)

    train_ds = LegalPairsDataset(train_records, tokenizer, mode=mode)
    dev_ds = LegalPairsDataset(dev_records, tokenizer, mode=mode)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_legal_batch(b, tokenizer),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_legal_batch(b, tokenizer),
    )

    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = (
        list(model.regression_head.parameters())
        + list(model.decision_head.parameters())
        + list(model.error_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    best_dev_rmse = float("inf")
    best_state = None

    print(f"Beginning training for {epochs} epochs ({len(train_records)} train samples)...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_scores = batch["lmp_score"].to(device)
            target_decisions = batch["decision_id"].to(device)
            target_errors = batch["error_vec"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask)

            loss_reg = mse_criterion(out["lmp_scores"], target_scores)
            loss_dec = ce_criterion(out["decision_logits"], target_decisions)
            loss_err = bce_criterion(out["error_logits"], target_errors)

            loss = loss_reg + 0.5 * loss_dec + 0.4 * loss_err
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        dev_metrics = evaluate_model(model, dev_loader, device)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Dev RMSE: {dev_metrics['rmse']:.4f} | Dev Pearson r: {dev_metrics['pearson_r']:.4f} | "
            f"Dev Acc: {dev_metrics['decision_acc']*100:.1f}%",
            flush=True,
        )

        if dev_metrics["rmse"] < best_dev_rmse:
            best_dev_rmse = dev_metrics["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with Dev RMSE = {best_dev_rmse:.4f}", flush=True)

    return model, tokenizer


if __name__ == "__main__":
    print("Testing multilingual_judgebert architecture...")
    model = MultilingualJUDGEBERT()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,} | Trainable parameters: {trainable:,}")
    print("Architecture initialized successfully!")
