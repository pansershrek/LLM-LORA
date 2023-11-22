import argparse
import json

import torch
from peft import PeftModel, prepare_model_for_kbit_training
import transformers
import pandas as pd

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, GenerationConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from sklearn.metrics import classification_report

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


ENTITIES = ["PER", "ORG", "LOC", "MISC"]
ENTITIES2ID = {y: x for x, y in enumerate(ENTITIES+["NONE"])}
ID2ENTITIES = {x: y for x, y in enumerate(ENTITIES+["NONE"])}

def check(x):
    if not x:
        return True
    if "none" in x.lower():
        return True
    if "not specified in the given input" in x.lower():
        return True
    if "n/a" in x.lower():
        return True
    if "not specified" in x.lower():
        return True
    if "not present in this input" in x.lower():
        return True
    if "not applicable in this context" in x.lower():
        return True
    if "not present in the given input" in x.lower():
        return True
    if "no entities of type" in x.lower():
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_weights", default="/home/admin/LLM-LORA/mistral_finetune_4")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--data_path", default="/home/admin/LLM-LORA/data/val_data.json")
    parser.add_argument("--output_path", default="model_outputs_4.csv")
    parser.add_argument("--output_f1_path", default="model_output_f1_4.txt")
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
        model = PeftModel.from_pretrained(model, args.lora_weights)
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

    for idx, data in enumerate(val_data):
        print(f"Step {idx}", flush=True)
        prompt = generate_prompt(
            data["instruction"], data["input"]
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda:0")
        generation_config = GenerationConfig.from_pretrained(args.model_name)

        #generation_config.renormalize_logits = True
        #whitelist = (
        #    [data["input"].split(" ")] +
        #    ["PER:", "ORG:", "LOC:", "MISC:"] +
        #    ["PER:\n", "ORG:\n", "LOC:\n", "MISC:\n", "\n"]
        #)
        #whitelist_ids = [tokenizer.encode(word)[0] for word in whitelist]
        #bad_words_ids=[[id] for id in range(tokenizer.vocab_size) if id not in whitelist_ids]


        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
                #bad_words_ids=bad_words_ids
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s).split("### Response:")[1].strip()
        ans = {
            "ground_truth": data["raw_entities"],
            "predict": {x: [] for x in ENTITIES},
            "raw_output": output,
            "input": data["input"],
            "full_prompt": prompt,
        }
        for k, v in ans["ground_truth"].items():
            ans["ground_truth"][k] = [y.lower() for y in v]
        output_list = output.split("\n")
        for entity in ENTITIES:
            for row in output_list:
                if entity+":" in row:
                    tmp_row = row.split(entity+":")[-1]
                    tmp_row = [y.strip().replace("</s>", "") for y in tmp_row.split(",")]
                    tmp_row = [y.lower() for y in tmp_row if not check(y)]
                    ans["predict"][entity] = tmp_row
                    break

        tmp_p = []
        tmp_g = []

        for y in ans["input"].split(" "):
            tmp_g.append(ENTITIES2ID["NONE"])
            tmp_p.append(ENTITIES2ID["NONE"])
            for entity in ENTITIES:
                if y.lower() in ans["ground_truth"][entity]:
                    tmp_g[-1] = ENTITIES2ID[entity]
                    break
            for entity in ENTITIES:
                if y.lower() in ans["predict"][entity]:
                    tmp_p[-1] = ENTITIES2ID[entity]
                    break
        ans["ground_truth_ner_classes"] = tmp_g
        ans["predict_ner_classes"] = tmp_p

        answer.append(ans)

    df = pd.DataFrame.from_dict(answer)
    df.to_csv(args.output_path)

    ground_truth = []
    predict = []
    for x in answer:
        ground_truth += x["ground_truth_ner_classes"]
        predict += x["predict_ner_classes"]

    with open(args.output_f1_path, "w") as f:
        print(
            classification_report(
                ground_truth, predict,
                target_names=['PER', 'ORG', 'LOC', 'MISC', 'NONE']
            ), file=f
        )


if __name__ == "__main__":
    main()