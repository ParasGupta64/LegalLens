# REF: https://huggingface.co/spaces/zoebat20/BLiMP/blob/main/app.py

import os
from functools import partial

import torch
import wandb
from datasets import load_dataset, DatasetDict
from dotenv import dotenv_values
from transformers import logging, pipeline

from factory import model_tokenizer_factory
from memory_cleanup import cleanup_memory
from src.metrics.metrics import (
    compute_metrics,
    eval_compute_metrics_identical,
    eval_compute_metrics_unrelated,
)

BASELINES = ["Aléatoire"]
BASELINES_FR = BASELINES + ["Annotateurs"]

# All LLM we want to evaluate
model_names = [
    "unsloth/llama-3-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "microsoft/Phi-3-mini-4k-instruct",
    "unsloth/gemma-2-2b-it-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "mistralai/Ministral-8B-Instruct-2410",
    "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
]

secrets = dotenv_values(".env")

huggingface_token = secrets["huggingface_token"]

device = torch.device("cuda")

# To make Wandb silent
os.environ["WANDB_SILENT"] = "true"

logging.set_verbosity_warning()

max_length = 2056

prompt = (
    "You will act as a legal advisor for laymen in Quebec. Your task is to classify if a tuple of sentences of a "
    "complex legal sentence that has been simplified conveys the same legal meaning. The legal meaning measures "
    "how well the output text coveys the legal details and exceptions and does not misrepresent the law. "
    "A score between 6 and 10 means the simplification seems to entail the legal details and exceptions "
    "properly and does not misrepresent the law, it is considered accurate. A score between 2 and 5 means "
    "the simplified sentence seems to entail the legal details and exceptions improperly and slightly misrepresents "
    "the law. A score of 1 means the simplification is completely off track. You can remove one point for each "
    "hallucination, omission or consistency between the original text and the simplification. We refer to "
    "hallucinations as facts included in the simplification that do not appear in the original text, omissions "
    "as cases where essential facts in the original are not in the simplified text, and consistency as "
    "inconsistency in the vocabulary used."
    "Original text: {}"
    "Simplified text: {}"
)


def evaluation_llm_prompting(rows, tokenizer, model):
    with torch.no_grad():
        original = rows["original"]
        simplified = rows["simplified"]

        formated_prompts = [prompt.format(original, simplified)]

        pipe = pipeline(model=model, tokenizer=tokenizer)
        output = pipe(
            formated_prompts,
            candidate_labels=[f"{i}" for i in range(1, 11)],
        )
        scores = output["scores"]
        prediction = scores.argmax()

        return {"prediction": prediction}


for data_augmentation in [False, True]:
    for model_name in model_names:
        batch_size = 128

        wandb.init(
            project=f"kil_llm_eval",
            config={"model_name": model_name},
        )
        clean_model_name = model_name.split("/")[-1]
        wandb.run.name = f"{clean_model_name}"

        model, tokenizer = model_tokenizer_factory(
            model_name=model_name,
            token=huggingface_token,
        )

        evaluation_fn = partial(
            evaluation_llm_prompting,
            tokenizer=tokenizer,
            model=model,
        )
        map_params = {"batched": True, "batch_size": batch_size}

        root_dir = os.path.join("..", "datastore", "training")
        if data_augmentation:
            train = load_dataset(
                root_dir,
                data_files=["train_FrJUDGE_with_data_augmentation.jsonl"],
            )
            dev = load_dataset(
                root_dir,
                data_files=["dev_FrJUDGE_with_data_augmentation.jsonl"],
            )
            test = load_dataset(
                root_dir,
                data_files=["test_FrJUDGE_with_data_augmentation.jsonl"],
            )
        else:
            train = load_dataset(
                root_dir,
                data_files=["train_FrJUDGE.jsonl"],
            )
            dev = load_dataset(
                root_dir,
                data_files=["dev_FrJUDGE.jsonl"],
            )
            test = load_dataset(
                root_dir,
                data_files=["test_FrJUDGE.jsonl"],
            )

        legal_precision_annotation_dataset = DatasetDict(
            {"train": train["train"], "dev": dev["train"], "test": test["train"]}
        )
        holdout_identical_dataset = load_dataset(
            root_dir, data_files=["holdout_identical_FrJUDGE.jsonl"]
        )
        holdout_unrelated_dataset = load_dataset(
            root_dir, data_files=["holdout_unrelated_FrJUDGE.jsonl"]
        )

        process_dataset = legal_precision_annotation_dataset.map(
            evaluation_fn,
            desc=f"----Doing model {model_name}-----",
            **map_params,
        )
        prediction = process_dataset["train"]["prediction"]
        labels = legal_precision_annotation_dataset["train"]["labels"]
        payload = {f"test/{k}": v for k, v in compute_metrics(zip(prediction, labels))}

        process_dataset = holdout_identical_dataset.map(
            evaluation_fn,
            desc=f"----Doing model {model_name}-----",
            **map_params,
        )
        prediction = process_dataset["train"]["prediction"]
        labels = holdout_identical_dataset["train"]["labels"]
        payload.update(
            {
                f"test/{k}": v
                for k, v in eval_compute_metrics_identical(zip(prediction, labels))
            }
        )

        process_dataset = holdout_unrelated_dataset.map(
            evaluation_fn,
            desc=f"----Doing model {model_name}-----",
            **map_params,
        )
        prediction = process_dataset["train"]["prediction"]
        labels = holdout_unrelated_dataset["train"]["labels"]
        payload.update(
            {
                f"test/{k}": v
                for k, v in eval_compute_metrics_unrelated(zip(prediction, labels))
            }
        )

        wandb.log({model_name: {**payload}})

        # We close the run since we will start a new one in the for loop for the next model.
        wandb.finish(exit_code=0)

        cleanup_memory(model=model, tokenizer=tokenizer)
