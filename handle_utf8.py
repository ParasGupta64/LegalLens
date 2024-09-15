# We use load_dataset from HG datasets library since it handle well JSONL
import os

from datasets import load_dataset

annotated_data = load_dataset(
    os.path.join("datastore", "annotated"),
    data_files=["insurance_text_simplifications_annotated.jsonl"],
)

annotated_data["train"].to_json(
    os.path.join(
        "datastore", "annotated", "insurance_text_simplifications_annotated.jsonl"
    ),
    force_ascii=False,
)
