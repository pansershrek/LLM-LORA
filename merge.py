import argparse
import json

from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="/data/LLM-LORA/config/openhermes_conll2003.json"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.loads(f.read())

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
    tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        config["MODEL_NAME"],
        device_map = "cpu"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = config["TRAIN_PARAMS"]["MAX_LEN"]
    model = PeftModel.from_pretrained(model, config["MODEL_OUTPUT"])
    model = model.merge_and_unload()

    model.save_pretrained(config["MERGED_MODEL_PATH"])
    tokenizer.save_pretrained(config["MERGED_MODEL_PATH"])


if __name__ == "__main__":
    main()
