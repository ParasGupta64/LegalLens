import os.path
import re

import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from evaluate import load
from tqdm import tqdm


def clean_columns(dataset):
    # We keep the original, the simplified, the difficulty evaluation (difficulty_evaluation),
    # the qualification class ("accept"), the legal difficulty level (evaluation),
    # and the justifications.
    dataset = dataset["train"].remove_columns(
        [
            "meta",
            "_input_hash",
            "_task_hash",
            "options",
            "text",
            "simplified_text",
            "_view_id",
            "config",
            "answer",
            "_timestamp",
            "_annotator_id",
            "_session_id",
        ]
    )

    # We convert to Pandas since it is easier to change the columns order
    df = dataset.to_pandas()
    # We reorder the column in the same logic as the article
    df = df.reindex(
        columns=[
            "original",
            "simplified",
            "difficulty_evaluation",
            "accept",
            "evaluation",
            "justifications",
        ]
    )
    # We rename the column to be aligned with the article
    df.columns = [
        "original",
        "simplification",
        "simplicity",
        "qualification",
        "legal_meaning",
        "justifications",
    ]
    dataset = Dataset.from_pandas(df)

    return dataset


def clean_columns_type(dataset):
    # We clean the qualification that is a list of a single element
    dataset = dataset.map(lambda x: {"qualification": x["qualification"][0]})
    return dataset


def create_training_corpus(dataset):
    # We create a train version with only the legal meaning criteria
    dataset = dataset.remove_columns(["simplicity", "qualification", "justifications"])
    # it is easier to just use "label" with the default HuggingFace training setup.
    dataset = dataset.rename_column("legal_meaning", "label")
    dataset = dataset.map(lambda x: {"label": float(x["label"])})

    # We use Pandas to group the evaluation and compute and average scores.
    df = dataset.to_pandas()
    df = (
        df.groupby(["original", "simplification"])["label"]
        .mean()
        .round()
        .to_frame()
        .reset_index()
    )

    dataset = Dataset.from_pandas(df)
    return dataset


def create_splits(dataset, seed: int = 42):
    # We create the splits
    # 60-10-30 train-test ratio
    # 30% for test
    trainvalid_test = dataset.train_test_split(
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
    return train_set, dev_set, test_set


def create_identical_pairs_data_augmentation_dataset(dataset):
    """
    Function to create identical pairs data augmentation dataset using the original as
    the original AND the simplification, thus the label is 10.0 (the maximum number).
    """
    sentence_pairs = []
    for row in tqdm(
        dataset.iter(batch_size=1), desc="Identical pairs creation", total=len(dataset)
    ):
        sentence_pairs.append(
            {
                "original": row["original"],
                "simplification": row["original"],
                "label": 10.0,
            }
        )

    identical_df = pd.DataFrame.from_dict(sentence_pairs, orient="columns")
    return identical_df


def create_unrelated_pairs_data_augmentation_dataset(dataset):
    """
    todo: change load to the official dataset when article is published
    """

    rouge = load("rouge")
    bleu = load("sacrebleu")

    datastore_root_path = os.path.join("..", "datastore", "references")

    quebec_insurance_legislature = load_dataset(
        datastore_root_path, data_files=["A-25.jsonl"]
    )
    quebec_traffic_legislature = load_dataset(
        datastore_root_path, data_files=["code_securite_routiere.jsonl"]
    )
    unrelated_sentences = concatenate_datasets(
        [quebec_insurance_legislature["train"], quebec_traffic_legislature["train"]]
    )

    unrelated_pairs = []
    running_idx = 0
    resampled = 0

    for row in tqdm(
        dataset.iter(batch_size=1), desc="Unrelated pairs creation", total=len(dataset)
    ):
        valid = False
        while not valid:
            original = row["original"]
            selected_unrelated_simplification = unrelated_sentences["text"][running_idx]
            # We clean the pattern of type "1. [...]" or "12.1 [...]" since it is not a common way top write.
            selected_unrelated_simplification = re.sub(
                pattern=r"[0-9]+\.*[0-9]*\. +",
                repl="",
                string=selected_unrelated_simplification,
            )
            # We also clean the new line and carriage return
            selected_unrelated_simplification = (
                selected_unrelated_simplification.replace("\n", " ")
            )
            selected_unrelated_simplification = (
                selected_unrelated_simplification.replace("\r", " ")
            )

            # We use the ROUGE and BLEU score to evaluate if the content seems related or not (in terms of vocabulary
            # not meaning).
            rouge_score = rouge.compute(
                predictions=[selected_unrelated_simplification], references=[original]
            )
            bleu_score = bleu.compute(
                predictions=[selected_unrelated_simplification], references=[original]
            )["score"]

            if (
                rouge_score["rouge1"] < 0.20
                and rouge_score["rouge2"] < 0.20
                and rouge_score["rougeL"] < 0.20
                and bleu_score < 20
            ):
                unrelated_pairs.append(
                    {
                        "original": original,
                        "simplification": selected_unrelated_simplification,
                        "label": 0.0,  # 0 since they are not related thus do not have the same legal meaning
                    }
                )
                valid = True
                running_idx += 1
            else:
                running_idx += 1
                resampled += 1
    unrelated_dataset_df = pd.DataFrame.from_dict(unrelated_pairs, orient="columns")
    return unrelated_dataset_df
