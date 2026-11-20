import os

from datasets import load_dataset, DatasetDict


def dataset_factory(data_augmentation: bool, dataset_name: str = "Multilingual_JUDGE", root_dir: str = None):
    if root_dir is None:
        # Automatically discover datastore directory
        candidates = [
            os.path.join("datastore", "training"),
            os.path.join("..", "..", "datastore", "training"),
            os.path.join(".", "datastore", "training"),
        ]
        for c in candidates:
            if os.path.exists(c):
                root_dir = c
                break
        if root_dir is None:
            root_dir = os.path.join("datastore", "training")

    prefix = "with_data_augmentation" if data_augmentation else None
    
    if data_augmentation:
        train_file = f"train_{dataset_name}_with_data_augmentation.jsonl"
        dev_file = f"dev_{dataset_name}_with_data_augmentation.jsonl"
        test_file = f"test_{dataset_name}_with_data_augmentation.jsonl"
    else:
        train_file = f"train_{dataset_name}.jsonl"
        dev_file = f"dev_{dataset_name}.jsonl"
        test_file = f"test_{dataset_name}.jsonl"

    train = load_dataset(root_dir, data_files=[train_file])
    dev = load_dataset(root_dir, data_files=[dev_file])
    test = load_dataset(root_dir, data_files=[test_file])

    legal_precision_annotation_dataset = DatasetDict(
        {"train": train["train"], "dev": dev["train"], "test": test["train"]}
    )
    holdout_identical_dataset = load_dataset(
        root_dir, data_files=[f"holdout_identical_{dataset_name}.jsonl"]
    )
    holdout_unrelated_dataset = load_dataset(
        root_dir, data_files=[f"holdout_unrelated_{dataset_name}.jsonl"]
    )

    return (
        legal_precision_annotation_dataset,
        holdout_identical_dataset,
        holdout_unrelated_dataset,
    )

