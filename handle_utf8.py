# We use load_dataset from HG datasets library since it handle well JSONL
annotated_data = load_dataset(
    "datastore", data_files=["pretest_annotated_dataset.jsonl"]
)

annotated_data["train"].to_json("test.jsonl", force_ascii=False)
