import os
from functools import partial

import numpy as np
import torch
import wandb
from dotenv import dotenv_values

from factory_dataset import dataset_factory
from evaluate_tools import evaluation_llm_prompting
from factory_model_tokenizer import model_tokenizer_factory
from metrics.metrics import (
    compute_metrics,
    eval_compute_metrics_identical,
    eval_compute_metrics_unrelated,
)

# To make Wandb silent
os.environ["WANDB_SILENT"] = "true"


def evaluation_loop(
    model_name: str,
    data_augmentation: bool,
    device_id: str = "0",
):
    secrets = dotenv_values(".env")

    huggingface_token = secrets["huggingface_token"]

    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = torch.device("cuda")

    wandb.init(
        project=f"kil_llm_eval",
        config={"model_name": model_name, "data_augmentation": data_augmentation},
        name=f"evaluate_LLM_with_DA-{data_augmentation}",
    )
    clean_model_name = model_name.split("/")[-1]

    model, tokenizer = model_tokenizer_factory(
        model_name=model_name,
        token=huggingface_token,
        device=device,
    )

    evaluation_fn = partial(
        evaluation_llm_prompting,
        tokenizer=tokenizer,
        model=model,
    )

    (
        legal_precision_annotation_dataset,
        holdout_identical_dataset,
        holdout_unrelated_dataset,
    ) = dataset_factory(data_augmentation=data_augmentation)

    process_dataset = legal_precision_annotation_dataset["test"].map(
        evaluation_fn,
        desc=f"----Doing model {model_name}-----",
    )

    prediction = np.array(process_dataset["prediction"])
    labels = legal_precision_annotation_dataset["test"]["label"]
    payload = {
        f"test/{clean_model_name}_{k}": v
        for k, v in compute_metrics((prediction, labels)).items()
    }

    process_dataset = holdout_identical_dataset.map(
        evaluation_fn,
        desc=f"----Doing model {model_name}-----",
    )
    prediction = np.array(process_dataset["train"]["prediction"])
    labels = holdout_identical_dataset["train"]["label"]
    payload.update(
        {
            f"test/{clean_model_name}_{k}": v
            for k, v in eval_compute_metrics_identical((prediction, labels)).items()
        }
    )

    process_dataset = holdout_unrelated_dataset.map(
        evaluation_fn,
        desc=f"----Doing model {model_name}-----",
    )
    prediction = np.array(process_dataset["train"]["prediction"])
    labels = holdout_unrelated_dataset["train"]["label"]
    payload.update(
        {
            f"test/{clean_model_name}_{k}": v
            for k, v in eval_compute_metrics_unrelated((prediction, labels)).items()
        }
    )

    wandb.log({**payload})

    # We close the run since we will start a new one in the for loop for the next model.
    wandb.finish(exit_code=0)
