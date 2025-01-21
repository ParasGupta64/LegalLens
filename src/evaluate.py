import argparse

from tools import bool_parse

from evaluation_loop import evaluation_loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to process the evaluation loop on.",
    )

    parser.add_argument(
        "data_augmentation",
        type=bool_parse,
        default=True,
        help="Whether to compute use the data augmentation dataset.",
    )

    parser.add_argument(
        "--device_id",
        type=str,
        default="0",
        help="Device ID tu use to be visible by CUDA.",
    )

    args = parser.parse_args()

    model_name = args.model_name
    data_augmentation = args.data_augmentation
    device_id = args.device_id

    evaluation_loop(
        model_name=model_name,
        data_augmentation=data_augmentation,
        device_id=device_id,
    )
