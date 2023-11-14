import argparse
import json

import torch
from peft import PeftModel, prepare_model_for_kbit_training
import transformers
import pandas as pd

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, GenerationConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


ENTITES = ["PER", "ORG", "LOC", "MISC"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_weights", default="/home/admin/LLM-LORA/mistral_finetune")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--data_path", default="/home/admin/LLM-LORA/data/val_data.json")
    parser.add_argument("--output_path", default="model_outputs.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map='cuda:0',
            use_flash_attention_2=True
        )
        model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        print("Work only with cuda")
        exit(0)

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    with open(args.data_path, "r") as f:
        val_data = json.loads(f.read())

    answer = []

    entity_correct_cnt = {x: 0 for x in ENTITES}
    entity_cnt = {x: 0 for x in ENTITES}

    for data in val_data:
        prompt = generate_prompt(
            data["instruction"], data["input"]
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig.from_pretrained(args.model_name)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s).split("### Response:")[1].strip()
        ans = {
            "ground_truth": data["raw_entities"],
            "predict": {},
            "rought_output": output
        }
        output_list = output.split("\n")
        for entity in ENTITES:
            for row in output_list:
                if row.startswith(entity+":"):
                    tmp_row = row.split(entity+":")[-1]
                    tmp_row = [x.strip().replace("</s>", "") for x in tmp_row.split(",")]
                    ans["predict"][entity] = tmp_row
                    break

        for entity in ENTITES:
            entity_cnt[entity] += len(data["raw_entities"][entity])
            for x, y in zip(
                sorted(data["raw_entities"][entity]),
                sorted(data["predict"][entity])
            ):
                if x.strip().lower() == y.strip().lower():
                    entity_correct_cnt[entity] += 1

        answer.append(ans)

    for entity in ENTITES:
        accuracy = 0
        if entity_cnt[entity]:
            accuracy = entity_correct_cnt[entity] / entity_cnt[entity] *  100.0
        print(f"Accuracy for {entity} entity is {accuracy:.2f}")

    df = pd.DataFrame.from_dict(answer)
    df.to_csv(args.output_path)


if __name__ == "__main__":
    main()