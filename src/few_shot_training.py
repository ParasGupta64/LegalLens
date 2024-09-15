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

from metrics.metrics import compute_metrics
from tools import (
    bool_parse,
)

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)

num_epoch = 5

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=45,
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
    default=True,
    help="Either or not to do data augmentation.",
)

args = parser.parse_args()

seed = args.seed
root = args.root
data_augmentation = args.data_augmentation

set_seeds(seed=seed)

# if data_augmentation:
legal_precision_annotation_dataset = load_dataset(
    os.path.join("datastore", "annotated"),
    data_files=["insurance_text_simplifications_annotated.jsonl"],
)

# 60-10-30 train-test ratio
# 30% for test
trainvalid_test = legal_precision_annotation_dataset.train_test_split(
    test_size=0.3, shuffle=True, seed=seed
)  # 70 is the 60-10 set for train/dev.
# Split the train in 60-10
train_valid = trainvalid_test["train"].train_test_split(
    test_size=0.1, shuffle=True, seed=seed
)

# The train set is the "train" set in the train_valid
# The dev set is the "test" set in the train_valid
# The test set is the "test" set in the trainvalid_test
train_set = train_valid["train"]
dev_set = train_valid["test"]
test_set = trainvalid_test["test"]

legal_precision_annotation_dataset = DatasetDict(
    {"train": train_set, "dev": dev_set, "test": test_set}
)

# else:
#     csmd_dataset = load_dataset("davebulaval/CSMD", "meaning")

# holdout_identical_dataset = load_dataset("davebulaval/CSMD", "meaning_holdout_identical")
# holdout_unrelated_dataset = load_dataset("davebulaval/CSMD", "meaning_holdout_unrelated")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(
        example["original"], example["simplification"], truncation=True, padding=True
    )


tokenized_legal_annotation_dataset = legal_precision_annotation_dataset.map(
    tokenize_function, batched=True
)
# tokenize_holdout_identical_dataset = holdout_identical_dataset.map(tokenize_function, batched=True)
# tokenize_holdout_unrelated_dataset = holdout_unrelated_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="legal_meaning_bert_train",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
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
    {"data_augmentation": f"{str(data_augmentation)} {num_epoch} holdout fixed"}
)
wandb.log({"Best model checkpoint path": trainer.state.best_model_checkpoint})

print("----------Test Set Evaluation start----------")
trainer.evaluate(
    eval_dataset=tokenized_legal_annotation_dataset["test"], metric_key_prefix="test"
)

# We change the computed metrics for the holdout splits
# trainer.compute_metrics = eval_compute_metrics_identical
# trainer.evaluate(
#     eval_dataset=tokenize_holdout_identical_dataset["test"],
#     metric_key_prefix="test/identical_sentences",
# )

# trainer.compute_metrics = eval_compute_metrics_unrelated
# trainer.evaluate(
#     eval_dataset=tokenize_holdout_unrelated_dataset["test"],
#     metric_key_prefix="test/unrelated_sentences",
# )

trainer.save_model(f"legal_meaningbert_best_model_{seed}")
