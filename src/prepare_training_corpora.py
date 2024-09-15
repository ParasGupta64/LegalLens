import os

from datasets import load_dataset, concatenate_datasets

from src.dataset_tools import (
    clean_columns,
    clean_columns_type,
    create_training_corpus,
    create_splits,
    create_identical_pairs_data_augmentation_dataset,
    create_unrelated_pairs_data_augmentation_dataset,
    create_holdout_dataset,
)

seed = 42

dataset = load_dataset(
    os.path.join("..", "datastore", "annotated"),
    data_files=["insurance_text_simplifications_annotated.jsonl"],
)

dataset = clean_columns(dataset)
dataset = clean_columns_type(dataset)

# We create the release dataset with all the columns
dataset.to_json(
    os.path.join("../datastore", "released", "insurance_text_simplifications.jsonl"),
    force_ascii=False,
)

dataset = create_training_corpus(dataset)

train_set, dev_set, test_set = create_splits(dataset, seed=seed)

train_set.to_json(
    os.path.join(
        "..", "datastore", "training", "train_insurance_text_simplifications.jsonl"
    ),
    force_ascii=False,
)
dev_set.to_json(
    os.path.join(
        "..", "datastore", "training", "dev_insurance_text_simplifications.jsonl"
    ),
    force_ascii=False,
)
test_set.to_json(
    os.path.join(
        "..", "datastore", "training", "test_insurance_text_simplifications.jsonl"
    ),
    force_ascii=False,
)

identical_pairs_data_augmentation_dataset = (
    create_identical_pairs_data_augmentation_dataset(dataset)
)
unrelated_pairs_data_augmentation_dataset, running_idx = (
    create_unrelated_pairs_data_augmentation_dataset(dataset)
)

dataset = concatenate_datasets(
    [
        dataset,
        identical_pairs_data_augmentation_dataset,
        unrelated_pairs_data_augmentation_dataset,
    ]
)

train_set.to_json(
    os.path.join(
        "..",
        "datastore",
        "training",
        "train_insurance_text_simplifications_with_data_augmentation.jsonl",
    ),
    force_ascii=False,
)
dev_set.to_json(
    os.path.join(
        "..",
        "datastore",
        "training",
        "dev_insurance_text_simplifications_with_data_augmentation.jsonl",
    ),
    force_ascii=False,
)
test_set.to_json(
    os.path.join(
        "..",
        "datastore",
        "training",
        "test_insurance_text_simplifications_with_data_augmentation.jsonl",
    ),
    force_ascii=False,
)

holdout = create_holdout_dataset(running_idx=running_idx)
