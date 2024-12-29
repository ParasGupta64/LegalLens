from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from transformers import logging

from unsloth import FastLanguageModel

logging.set_verbosity_warning()


def model_tokenizer_factory(model_name, token):
    if (
        "bloom" in model_name.lower()
        or "moe" in model_name.lower()
        or "chocolatine" in model_name.lower()
        or "alpaca" in model_name.lower()
    ):
        bnb_configs = BitsAndBytesConfig(load_in_8bit=True, low_cpu_mem_usage=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            quantization_config=bnb_configs,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    elif "flan" in model_name.lower():
        bnb_configs = BitsAndBytesConfig(load_in_8bit=True, low_cpu_mem_usage=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            token=token,
            quantization_config=bnb_configs,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            token=token,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    return model, tokenizer
