import argparse
import os
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import transformers

def get_dataset(data_path):
    return load_dataset("json", data_files=data_path)["train"]

def main():
    test_dataset = get_dataset(
        "/mnt/data/g.skiba/LLM-LORA/dataset/conll2003_dataset_test.json"
    )

    model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
    )

    for x in test_dataset:
        prompt = pipeline.tokenizer.apply_chat_template(
            x["input"], tokenize=False, add_generation_prompt=True
        )
        outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        print(outputs[0]["generated_text"])
        break


if __name__ == "__main__":
    main()