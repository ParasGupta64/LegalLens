# How to annotate the dataset using Prodigy

Here is the command to self-host Prodigy

```shell
prodigy legal_meaning legal_annotations ./datastore/insurance_text_simplifications.jsonl -F recipe.py 
```

Output the annotation:

```shell
prodigy db-out legal_annotations > legal_annotations.jsonl
```