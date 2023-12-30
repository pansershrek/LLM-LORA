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
        device_map="auto"
    )
    ans = []
    for idx, x in enumerate(test_dataset):
        messages = [{"role": "user", "content": x["input"]}]
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = pipeline(
            prompt, max_new_tokens=800, temperature=0.75, top_p=0.9
        )
        if idx % 1000 == 0:
            print(outputs[0]["generated_text"], flush=True)
        x["generated_output"] = outputs[0]["generated_text"]
        ans.append(x)

    with open("/mnt/data/g.skiba/LLM-LORA/mixtral_generation_result.json", "w") as f:
        print(json.dumps(ans), file=f)


if __name__ == "__main__":
    main()