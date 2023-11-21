import os

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

#assert (
#    "LlamaTokenizer" in transformers._import_structure["models.llama"]
#), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
#from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


import json

import wandb
from peft import (
    LoraConfig, PeftConfig, PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig,
                          DataCollatorForSeq2Seq,
                          DataCollatorForTokenClassification, EvalPrediction,
                          T5ForConditionalGeneration, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


MODEL_OUTPUT = "mistral_finetune_4"


# optimized for RTX 3090 and A100. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 2  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't always need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 512  # 256 accounts for about 96% of the data
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj"
]

model_name = "mistralai/Mistral-7B-Instruct-v0.1" #"meta-llama/Llama-2-13b-hf" #"mistralai/Mistral-7B-Instruct-v0.1"
#model_name = "bigscience/bloom-1b1"
#model_name = "bigscience/bloom-1b7"
#model_name = "bigscience/bloom-3b"
#model_name = "bigscience/bloom-7b1"
#model_name = "bigscience/bloom" # for 176B parameters

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda:0',
    load_in_8bit=True,
    use_flash_attention_2=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_config = AutoConfig.from_pretrained(model_name)
#tokenizer = fix_tokenizer(tokenizer, model_config) # Maybe to comment
#model = fix_model(model, tokenizer, use_resize=False) # Maybe to comment

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.config.max_length = CUTOFF_LEN
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
#data = load_dataset("json", data_files=DATA_PATH)

#train_val = data["train"].train_test_split(
#    test_size=VAL_SET_SIZE, shuffle=True, seed=42
#)
train_data = load_dataset("json", data_files="data/train_data.json")["train"]
val_data = load_dataset("json", data_files="data/val_data.json")["train"]


def generate_prompt(data_point, without_system=False):
    # sorry about the formatting disaster gotta move fast
    if without_system and data_point["input"]:
        return f"""### Task: {data_point["instruction"]}

### Input: {data_point["input"]}

### Output: {data_point["output"]}"""
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point, without_system=True):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
        )
        if data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        )
    )
    if without_system:
        user_prompt = f"""### Task: {data_point["instruction"]}

### Input: {data_point["input"]}

### Output: """
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        )
        - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": full_tokens,
        #"labels": [-100] * len_user_prompt_tokens
        #+ full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
val_data = val_data.shuffle().map(generate_and_tokenize_prompt)

data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        report_to='wandb',
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        evaluation_strategy="epoch",
        #evaluation_strategy="epoch",
        #save_strategy="epoch",
        #eval_steps=200,
        save_steps=100,
        output_dir=MODEL_OUTPUT,
        #save_total_limit=3,
        #load_best_model_at_end=True,
    ),
    data_collator=data_collator,
)



model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2":
    model = torch.compile(model)

with wandb.init(project="Instruction NER") as run:
    model.print_trainable_parameters()
    trainer.train() #if resume, choose True, else False
    #torch.save(model.state_dict(), f"final_mistral_1.pth")

model.save_pretrained(MODEL_OUTPUT)

print("\n If there's a warning about missing keys above, please disregard :)")
