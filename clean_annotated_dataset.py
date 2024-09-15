import os

from datasets import load_dataset, Dataset

dataset = load_dataset(
    os.path.join("datastore", "annotated"),
    data_files=["insurance_text_simplifications_annotated.jsonl"],
)

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
# We clean the qualification that is a list of a single element
dataset = dataset.map(lambda x: {"qualification": x["qualification"][0]})

dataset.to_json(
    os.path.join("datastore", "released", "insurance_text_simplifications.jsonl"),
    force_ascii=False,
)

# We create a train version with only the legal meaning criteria
dataset = dataset.remove_columns(["simplicity", "qualification", "justifications"])
# it is easier to just use "label" with the default HuggingFace training setup.
dataset = dataset.rename_column("legal_meaning", "label")
dataset = dataset.map(lambda x: {"label": int(x["label"])})
dataset.to_json(
    os.path.join("datastore", "training", "insurance_text_simplifications.jsonl"),
    force_ascii=False,
)
