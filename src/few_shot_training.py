import argparse
import logging
import os

import wandb
from datasets import load_dataset, DatasetDict
from poutyne import set_seeds
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from metrics.metrics import (
    compute_metrics,
    eval_compute_metrics_identical,
    eval_compute_metrics_unrelated,
)
from tools import (
    bool_parse,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # or "0,1" for multiple GPUs

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="The seed to use for training.",
)

parser.add_argument(
    "--root",
    type=str,
    default=".",
    help="Root directory.",
)

parser.add_argument(
    "--data_augmentation",
    type=bool_parse,
    default=False,
    help="Either or not to do data augmentation.",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="The number of epochs to complete.",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="bert-base-uncased",
    help="The model name to use as the starting model checkpoint.",
)

args = parser.parse_args()

seed = args.seed
root = args.root
data_augmentation = args.data_augmentation
num_epoch = args.epochs

set_seeds(seed=seed)
root_dir = os.path.join("..", "datastore", "training")

if data_augmentation:
    train = load_dataset(
        root_dir,
        data_files=[
            "train_insurance_text_simplifications_with_data_augmentation.jsonl"
        ],
    )
    dev = load_dataset(
        root_dir,
        data_files=["dev_insurance_text_simplifications_with_data_augmentation.jsonl"],
    )
    test = load_dataset(
        root_dir,
        data_files=["test_insurance_text_simplifications_with_data_augmentation.jsonl"],
    )
else:
    train = load_dataset(
        root_dir,
        data_files=["train_insurance_text_simplifications.jsonl"],
    )
    dev = load_dataset(
        root_dir,
        data_files=["dev_insurance_text_simplifications.jsonl"],
    )
    test = load_dataset(
        root_dir,
        data_files=["test_insurance_text_simplifications.jsonl"],
    )

legal_precision_annotation_dataset = DatasetDict(
    {"train": train["train"], "dev": dev["train"], "test": test["train"]}
)

holdout_identical_dataset = load_dataset(
    root_dir, data_files=["holdout_identical_insurance_text_simplifications.jsonl"]
)
holdout_unrelated_dataset = load_dataset(
    root_dir, data_files=["holdout_unrelated_insurance_text_simplifications.jsonl"]
)

checkpoint = args.model_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(
        example["original"], example["simplification"], truncation=True, padding=True
    )


tokenized_legal_annotation_dataset = legal_precision_annotation_dataset.map(
    tokenize_function, batched=True
)
tokenize_holdout_identical_dataset = holdout_identical_dataset.map(
    tokenize_function, batched=True
)
tokenize_holdout_unrelated_dataset = holdout_unrelated_dataset.map(
    tokenize_function, batched=True
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=f"legal_meaning_bert_train_{checkpoint}",
    logging_strategy="epoch",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=num_epoch,
    save_total_limit=num_epoch,
    save_strategy="epoch",
    load_best_model_at_end=True,  # By default, use the eval loss to retrieve the best model.
    seed=seed,
    metric_for_best_model="eval_loss",
    learning_rate=5e-5,
)

# num_labels to 1 to create a regression head
# REF: https://discuss.huggingface.co/t/fine-tune-bert-and-camembert-for-regression-problem/332/17
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_legal_annotation_dataset["train"],
    eval_dataset=tokenized_legal_annotation_dataset["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("----------Training start----------")
trainer.train()

wandb.run.config.update(
    {
        "data_augmentation": f"{str(data_augmentation)} {num_epoch} holdout fixed",
        "tag": f"legal_meaning_bert_train_{checkpoint}",
    }
)
wandb.log({"Best model checkpoint path": trainer.state.best_model_checkpoint})

print("----------Test Set Evaluation start----------")
trainer.evaluate(
    eval_dataset=tokenized_legal_annotation_dataset["test"], metric_key_prefix="test"
)

# We change the computed metrics for the holdout splits
trainer.compute_metrics = eval_compute_metrics_identical
trainer.evaluate(
    eval_dataset=tokenize_holdout_identical_dataset["train"],
    metric_key_prefix="test/identical_sentences",
)

trainer.compute_metrics = eval_compute_metrics_unrelated
trainer.evaluate(
    eval_dataset=tokenize_holdout_unrelated_dataset["train"],
    metric_key_prefix="test/unrelated_sentences",
)
