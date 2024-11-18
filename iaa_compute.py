import json
import os

import pandas as pd
from datasets import load_dataset
from prodigy_iaa.measures import calculate_agreement

# We use load_dataset from HG datasets library since it handle well JSONL
annotated_data = load_dataset(
    os.path.join("datastore", "annotated"),
    data_files=["insurance_text_simplifications_annotated.jsonl"],
)

# Each annotation per annotator is a separated row
# "train" since by default HG set it as train
extracted_annotations = {}

difficulty_to_int_class = {
    key: value
    for key, value in zip(
        [
            "Plus simple à lire",
            "Aussi simple à lire",
            "Plus difficile à lire",
            "Pas de simplification",
        ],
        range(0, 4),
    )
}


def df_to_reliability(df):
    # Taken from https://github.com/pmbaumgartner/prodigy-iaa/
    # blob/a802f493e09473e6725367340dd316f44cd956d9/tests/conftest.py#L7C1-L12C28
    """Converts a DataFrame to a list of lists, where missing values
    are replaced with `None`."""
    df = df.where(pd.notnull(df), None)
    reliability_data = df.values.tolist()
    return reliability_data


def convert_name_to_unique_id(annotator):
    if "William" in annotator:
        unique_id = 1
    elif "Eve" in annotator:
        unique_id = 2
    elif "Simon" in annotator:
        unique_id = 3
    elif "Yanis" in annotator:
        unique_id = 4
    else:
        raise Exception("Unknown annotator")
    return unique_id


sentence_id_mapping = []
for row in annotated_data["train"]:
    # The level of difficulty amongst the preset class (7 classes)
    difficulty_level_annotation = difficulty_to_int_class.get(
        row["difficulty_evaluation"]
    )

    # The qualification of the original text (18 classes)
    qualification_annotation = int(row["accept"][0])

    # The legal evaluation level between 1 and 10
    legal_precision_annotation = int(row["evaluation"])

    justifications_annotation = row["justifications"]

    annotator = row["_session_id"]
    annotator_numerical_id = convert_name_to_unique_id(annotator)

    row_data = {
        "difficulty_level_annotation": difficulty_level_annotation,
        "qualification_annotation": qualification_annotation,
        "legal_precision_annotation": legal_precision_annotation,
        "justifications_annotation": justifications_annotation,
    }

    annotation_hash = row["_input_hash"]
    if extracted_annotations.get(annotation_hash) is None:
        extracted_annotations.update(
            {annotation_hash: {str(annotator_numerical_id): row_data}}
        )
    else:
        annotation_datas = extracted_annotations.get(annotation_hash)
        annotation_datas.update({str(annotator_numerical_id): row_data})
        extracted_annotations.update({annotation_hash: annotation_datas})

    original = row["original"]
    simplified = row["simplified"]
    sentence_id_mapping.append(
        {"sentence_id": annotation_hash, "original": original, "simplified": simplified}
    )

dataframes = []

tasks = ["difficulty_level", "qualification", "legal_precision"]  # , "justifications"]
for task in tasks:
    tasks_annotation = []
    for sentence_idx, sentence in extracted_annotations.items():
        annotations = []
        for annotator_annotations in sentence.values():
            annotations.append(annotator_annotations.get(f"{task}_annotation", ""))

        tasks_annotation.append([sentence_idx] + annotations)

    task_df = pd.DataFrame(
        tasks_annotation,
        columns=[
            "sentence_id",
            "Annotator A",
            "Annotator B",
            "Annotator C",
            "Annotator D",
        ],
    )

    # Since it will consider the sentence id as an evaluator
    agreement_df = task_df.drop("sentence_id", axis=1)

    agreement = calculate_agreement(df_to_reliability(agreement_df))
    print(agreement)
    with open(os.path.join("results", f"{task}_agreement.csv"), "w") as f:
        json.dump(agreement, f)

    dataframes.append((task, task_df, agreement))

for task, data, iaa in dataframes:
    data.replace("\n", "", regex=True, inplace=True)
    data.to_csv(
        os.path.join("results", f"insurance_text_simplifications_annotated_{task}.tsv"),
        index=False,
        sep="\t",
    )
    data.columns = ["sentence_id"] + [
        f"{column}-{task}" for column in data.columns if column != "sentence_id"
    ]

original_data = pd.DataFrame(sentence_id_mapping).drop_duplicates()

merge_df = pd.merge(left=original_data, right=dataframes[0][1])
merge_df = pd.merge(left=merge_df, right=dataframes[1][1])
merge_df = pd.merge(left=merge_df, right=dataframes[2][1])
# merge_df = pd.merge(left=merge_df, right=dataframes[3][1])

path = os.path.join("results", "insurance_text_simplifications_annotated.tsv")
merge_df.to_csv(path, index=False)
