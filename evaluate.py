import argparse
import json

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
        "--config", default="/data/LLM-LORA/config/mistral_conll2003.json"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.loads(f.read())

    model = vllm(config["MERGED_MODEL_PATH"])
    sampling_params = SamplingParams(
        temperature = config["SAMPLING_PARAMS"]["TEMPERATURE"],
        top_k = config["SAMPLING_PARAMS"]["TOP_K"],
        top_p = config["SAMPLING_PARAMS"]["TOP_P"],
        max_tokens = config["SAMPLING_PARAMS"]["MAX_TOKENS"],
    )

    test_dataset = get_dataset(config["TEST_DATASET"])
    batched_test_dataset = create_batched_dataset(
        test_dataset, 4
    )

    for x in batched_test_dataset:
        outputs = model.generate(x["input"], sampling_params)
        print(outputs.outputs)
        break

if __name__ == "__main__":
    main()