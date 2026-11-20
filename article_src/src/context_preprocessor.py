"""
context_preprocessor.py
Dual-Mode Context-Aware and Sentence-Only Data Preprocessor for JUDGEBERT.

Supports:
1. Sentence-Only mode: [CLS] original [SEP] simplification [SEP]
2. Context-Aware mode: [CLS] Context: {context} | Original: {original} [SEP] Simplified: {simplification} [SEP]
3. Multi-task target extraction:
   - Regression label (float 1.0 - 10.0)
   - Preservation Decision (0: Not Preserved, 1: Partially Preserved, 2: Preserved)
   - Multi-hot legal error tags (contradiction, omission, dates_numbers, condition, scope, exception, obligation, entity)
"""

import os
from typing import Dict, List, Optional
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

# Master mapping for 3-class Preservation Decisions
DECISION_TO_ID = {
    "Not Preserved": 0,
    "Partially Preserved": 1,
    "Preserved": 2
}
ID_TO_DECISION = {v: k for k, v in DECISION_TO_ID.items()}

# Master mapping for legal error taxonomy
ERROR_TAXONOMY = [
    "contradiction",
    "omission",
    "dates_numbers_shift",
    "condition_shift",
    "scope_shift",
    "exception_shift",
    "obligation_shift",
    "entity_shift",
    "hallucination"
]
ERROR_TO_ID = {err: i for i, err in enumerate(ERROR_TAXONOMY)}
ID_TO_ERROR = {i: err for i, err in enumerate(ERROR_TAXONOMY)}


def score_to_decision(score: float) -> str:
    """Classifies continuous LMP score into standard 3-tier preservation decision."""
    if score >= 8.0:
        return "Preserved"
    elif score >= 5.0:
        return "Partially Preserved"
    else:
        return "Not Preserved"


def get_error_multihot(error_types: List[str]) -> List[float]:
    """Encodes list of error strings into binary multi-hot float vector."""
    vec = [0.0] * len(ERROR_TAXONOMY)
    for err in error_types:
        if err in ERROR_TO_ID:
            vec[ERROR_TO_ID[err]] = 1.0
    return vec


def trim_context_for_clause(context: str, original: str, max_chars: int = 650) -> str:
    """
    Intelligently trims legal context so that it provides relevant section headers
    and surrounding language while preventing tokenizer truncation of the original clause.
    """
    if not context or len(context) <= max_chars:
        return context.strip()

    context_clean = context.strip()
    orig_clean = original.strip() if original else ""

    # If original clause (or significant snippet) is present in context, center window around it
    if orig_clean and len(orig_clean) >= 15:
        snippet = orig_clean[:min(45, len(orig_clean))].lower()
        idx = context_clean.lower().find(snippet)
        if idx != -1:
            half = max_chars // 2
            start = max(0, idx - half)
            end = min(len(context_clean), idx + half)
            if start > 0:
                space_idx = context_clean.find(" ", start)
                if space_idx != -1 and space_idx < idx:
                    start = space_idx + 1
            if end < len(context_clean):
                space_idx = context_clean.rfind(" ", idx, end)
                if space_idx != -1 and space_idx > idx:
                    end = space_idx
            
            trimmed = context_clean[start:end].strip()
            prefix = "... " if start > 0 else ""
            suffix = " ..." if end < len(context_clean) else ""
            return f"{prefix}{trimmed}{suffix}"

    # Otherwise take introductory/section headings up to max_chars
    trimmed = context_clean[:max_chars]
    last_punct = max(trimmed.rfind("."), trimmed.rfind("\n"), trimmed.rfind(";"), trimmed.rfind(":"))
    if last_punct > max_chars * 0.6:
        trimmed = trimmed[:last_punct + 1]
    else:
        last_space = trimmed.rfind(" ")
        if last_space > 0:
            trimmed = trimmed[:last_space]
    return f"{trimmed}..."


def format_input_text(example: Dict, mode: str = "sentence_only", max_context_chars: int = 650) -> tuple:
    """
    Returns (text_a, text_b) based on preprocessing mode.
    - sentence_only: (original, simplification)
    - context_aware: (Context: {context} | Original: {original}, Simplified: {simplification})
    """
    original = example.get("original") or example.get("original_text", "")
    simplification = example.get("simplification") or example.get("simplified_text", "")
    context = example.get("context_text") or example.get("context", "")

    if mode == "context_aware" and context:
        effective_context = trim_context_for_clause(context, original, max_chars=max_context_chars)
        text_a = f"Context: {effective_context} | Original Legal Clause: {original}"
        text_b = f"Simplified Candidate: {simplification}"
    else:
        # Default sentence-only mode
        text_a = original
        text_b = simplification

    return text_a, text_b


def create_tokenize_function(tokenizer, mode: str = "sentence_only", max_length: int = 512):
    """Factory creating Hugging Face dataset mapping function for tokenization."""
    def tokenize_fn(batch):
        texts_a = []
        texts_b = []

        num_examples = len(batch["original"]) if "original" in batch else len(batch["original_text"])
        
        for i in range(num_examples):
            ex = {k: batch[k][i] for k in batch.keys()}
            ta, tb = format_input_text(ex, mode=mode)
            texts_a.append(ta)
            texts_b.append(tb)

        tokenized = tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            padding=False,  # dynamically padded during collation
            max_length=max_length,
            return_token_type_ids=False
        )

        # Ensure label is float for PyTorch MSE/Regression loss
        if "label" in batch:
            tokenized["labels"] = [float(x) for x in batch["label"]]
        elif "lmp_score" in batch:
            tokenized["labels"] = [float(x) for x in batch["lmp_score"]]

        # Optional decision and error labels for multi-task evaluation
        if "decision" in batch:
            tokenized["decision_label"] = [DECISION_TO_ID.get(d, 0) for d in batch["decision"]]
            
        if "error_types" in batch:
            tokenized["error_labels"] = [get_error_multihot(errs) for errs in batch["error_types"]]

        return tokenized

    return tokenize_fn


def prepare_tokenized_dataset(
    dataset_dict: DatasetDict,
    tokenizer,
    mode: str = "sentence_only",
    max_length: int = 512
) -> DatasetDict:
    """
    Tokenizes all splits in a DatasetDict using the designated mode.
    """
    tokenize_fn = create_tokenize_function(tokenizer, mode=mode, max_length=max_length)
    
    tokenized = DatasetDict()
    for split_name, ds in dataset_dict.items():
        tokenized[split_name] = ds.map(
            tokenize_fn,
            batched=True,
            desc=f"Tokenizing {split_name} ({mode})"
        )
        
    return tokenized


if __name__ == "__main__":
    print("Testing context_preprocessor...")
    checkpoint = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    sample = {
        "original": "Tenant shall pay rent of $5,000 on the first day of each month.",
        "simplification": "Tenant must pay $5,000 monthly rent on the 1st.",
        "context_text": "LEASE AGREEMENT - Clause 4: Rent and Financial Obligations.",
        "lmp_score": 9.5,
        "label": 9.5,
        "decision": "Preserved",
        "error_types": ["none"]
    }

    # Test Sentence-Only
    ta_s, tb_s = format_input_text(sample, mode="sentence_only")
    enc_s = tokenizer(ta_s, tb_s)
    print("\n[Sentence-Only Mode]")
    print("Text A:", ta_s)
    print("Text B:", tb_s)
    print("Tokens length:", len(enc_s["input_ids"]))
    print("Decoded:", tokenizer.decode(enc_s["input_ids"]))

    # Test Context-Aware
    ta_c, tb_c = format_input_text(sample, mode="context_aware")
    enc_c = tokenizer(ta_c, tb_c)
    print("\n[Context-Aware Mode]")
    print("Text A:", ta_c)
    print("Text B:", tb_c)
    print("Tokens length:", len(enc_c["input_ids"]))
    print("Decoded:", tokenizer.decode(enc_c["input_ids"]))
    print("\nPreprocessor verified successfully!")
