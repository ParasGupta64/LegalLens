import os

import numpy as np
import pandas as pd
import plotly.express as px
import wandb
from tqdm import tqdm

saving_dir = "figs"

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs(f"davebulaval/huggingface")

evaluation_metrics_names = [
    "BERTScore precision",
    "BERTScore recall",
    "BERTScore F1",
    "QuestEval",
    "LENS",
    "Coverage",
    "MeaningBERT",
    "SBERT Embeddings",
    "SBERT Semantic Search",
    "SBERT Average Perf",
]

# Metadata
batch_with_da, batch_model_name = [], []

# Experiment results monitored for analysis purpose that we want to extract
batch_r2, batch_pearson_corr, batch_rmse = [], [], []
identical_batch_ratio_95, identical_batch_ratio_99 = [], []
unrelated_batch_ratio_5, unrelated_batch_ratio_1 = [], []
for run in tqdm(runs):
    if run.state == "finished":
        with_da = run.metadata.get("args")[1].split("=")[-1]

        if len(run.metadata.get("args")) > 3:
            # WE only specify the model when we use bert-large, otherwise we use the default value of "bert-base".
            model_name = run.metadata.get("args")[3].split("=")[-1]
        elif "evaluate_other_metrics" in run.name:
            # None since weh we evaluate other metrics, we do them in batch all of them, thus we have more than one
            # model name, and we will extract it later.
            model_name = None
        else:
            model_name = "bert-base-uncased"

        if model_name is not None:
            batch_with_da.append(with_da)
            batch_model_name.append(model_name)

            batch_r2.append(
                [v for k, v in run.summary._json_dict.items() if k == "test/R2"][0]
            )

            batch_pearson_corr.append(
                [
                    v
                    for k, v in run.summary._json_dict.items()
                    if k == "test/pearson_corr"
                ][0]
            )

            batch_rmse.append(
                [v for k, v in run.summary._json_dict.items() if k == "test/rmse"][0]
            )

            identical_batch_ratio_95.append(
                [
                    v
                    for k, v in run.summary._json_dict.items()
                    if k == "train/test/identical_sentences_ratio_95"
                ][0]
            )

            identical_batch_ratio_99.append(
                [
                    v
                    for k, v in run.summary._json_dict.items()
                    if k == "train/test/identical_sentences_ratio_99"
                ][0]
            )

            unrelated_batch_ratio_5.append(
                [
                    v
                    for k, v in run.summary._json_dict.items()
                    if k == "train/test/unrelated_sentences_ratio_5"
                ][0]
            )

            unrelated_batch_ratio_1.append(
                [
                    v
                    for k, v in run.summary._json_dict.items()
                    if k == "train/test/unrelated_sentences_ratio_1"
                ][0]
            )
        else:
            batch_with_da.extend([with_da] * len(evaluation_metrics_names))

            batch_model_name.extend(evaluation_metrics_names)

            for metric in evaluation_metrics_names:
                key_to_extract = f"test/{metric}_R2"
                value = run.summary._json_dict.get(key_to_extract, None)
                batch_r2.append(value)

                key_to_extract = f"test/{metric}_pearson_corr"
                value = run.summary._json_dict.get(key_to_extract, None)
                batch_pearson_corr.append(value)

                key_to_extract = f"test/{metric}_RMSE"
                value = run.summary._json_dict.get(key_to_extract, None)
                batch_rmse.append(value)

                key_to_extract = f"test/{metric}_ratio_95"
                value = run.summary._json_dict.get(key_to_extract, None)
                identical_batch_ratio_95.append(value)

                key_to_extract = f"test/{metric}_ratio_99"
                value = run.summary._json_dict.get(key_to_extract, None)
                identical_batch_ratio_99.append(value)

                key_to_extract = f"test/{metric}_ratio_1"
                value = run.summary._json_dict.get(key_to_extract, None)
                unrelated_batch_ratio_5.append(value)

                key_to_extract = f"test/{metric}_ratio_5"
                value = run.summary._json_dict.get(key_to_extract, None)
                unrelated_batch_ratio_1.append(value)

run_df = pd.DataFrame(
    {
        "With DA": batch_with_da,
        "Model Name": batch_model_name,
        "R2": batch_r2,
        "Pearson Corr": batch_pearson_corr,
        "RMSE": batch_rmse,
        "Identical 95": identical_batch_ratio_95,
        "Identical 99": identical_batch_ratio_99,
        "Unrelated 5": unrelated_batch_ratio_5,
        "Unrelated 1": unrelated_batch_ratio_1,
    },
)

run_df["color"] = np.where(
    run_df["model_name"] == "Aléatoire",
    "red",
    np.where(run_df["model_name"] == "Annotateurs", "black", "blue"),
)

fig = px.scatter(
    run_df,
    x="model_size",
    y="accuracy",
    color=run_df["color"],
    trendline="ols",
    trendline_options=dict(log_x=True),
    log_x=True,
    hover_name="model_name",
).update_layout(
    title=dict(
        text=f"{lang.upper()} minimal pair log-scaled X axis and log-transformed fit",
        font=dict(size=24, color="#000000"),
        y=1,
        x=0.5,
        xanchor="center",
        yanchor="top",
    ),
    xaxis_title="Model size",
    yaxis_title="Accuracy",
    showlegend=False,
)

# fig.update_layout(yaxis_range=[25, 90])

fig.add_hline(y=random_accuracy_value, line_width=1, line_dash="dash", line_color="red")
if lang == "fr":
    fig.add_hline(
        y=annotators_accuracy_value,
        line_width=1,
        line_dash="dash",
        line_color="green",
    )

fig.show(renderer="browser")

run_df.drop(["color"], axis=1, inplace=True)

run_df.to_csv(os.path.join("results", f"result_{lang}.tsv"), index=False, sep="\t")
fig.write_html(os.path.join("results", f"minimal_pair_analysis_{lang}.html"))
