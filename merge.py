import argparse
import json

import torch
from peft import PeftModel, prepare_model_for_kbit_training
import transformers
import pandas as pd

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, GenerationConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    #load_in_8bit=True,
    #torch_dtype=torch.float16,
    device_map='cpu'
)
#model = prepare_model_for_kbit_training(model)
model = PeftModel.from_pretrained(model, "/home/admin/LLM-LORA/mistral_finetune_4")
model = model.merge_and_unload()
model.save_pretrained("/data/LLM-LORA/new_weights")