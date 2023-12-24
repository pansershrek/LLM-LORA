import argparse
import os
import json

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

def get_dataset(data_path):
    return load_dataset("json", data_files=data_path)["train"]

def create_batched_dataset(dataset, batch_size):
    ans = []
    tmp = {
        "output": [],
        "input": [],
        "converted_ner_tags": []
    }
    for x in dataset:
        if len(tmp["output"]) == batch_size:
            ans.append(tmp)
            tmp = {
                "output": [],
                "input": [],
                "converted_ner_tags": []
            }
        tmp["output"].append(x["output"])
        tmp["input"].append(x["input"])
        tmp["converted_ner_tags"].append(x["converted_ner_tags"])

    if tmp:
        ans.append(tmp)
    return ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="/data/LLM-LORA/config/openhermes_conll2003.json"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.loads(f.read())
    if "ONLY_EVAL" in config and config["ONLY_EVAL"]:
        model = LLM(config["MODEL_NAME"])
    else:
        model = LLM(
            config["MERGED_MODEL_PATH"]
            #tensor_parallel_size = torch.cuda.device_count()
        )
    sampling_params = SamplingParams(
        temperature = config["SAMPLING_PARAMS"]["TEMPERATURE"],
        top_k = config["SAMPLING_PARAMS"]["TOP_K"],
        top_p = config["SAMPLING_PARAMS"]["TOP_P"],
        max_tokens = config["SAMPLING_PARAMS"]["MAX_TOKENS"],
    )

    test_dataset = get_dataset(config["TEST_DATASET"])
    batched_test_dataset = create_batched_dataset(
        test_dataset, config["SAMPLING_PARAMS"]["BATCH_SIZE"]
    )

    generated_text = []
    for idx, x in enumerate(batched_test_dataset):
        print(
            f"Step {idx * config['SAMPLING_PARAMS']['BATCH_SIZE']}",
            flush=True
        )
        outputs = model.generate(x["input"], sampling_params)
        texts = [x.outputs[0].text for x in outputs]
        generated_text += texts

    ans = []
    for x, y in zip(test_dataset, generated_text):
        x["generated_text"] = y
        ans.append(x)

    with open(
        os.path.join(config["MODEL_OUTPUT"], "generated_output.json"), "w"
    ) as f:
        print(json.dumps(ans, indent=4), file=f)


if __name__ == "__main__":
    main()