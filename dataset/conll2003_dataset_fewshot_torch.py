import json
import copy
import random
import time

FEW_SHOT = 3

with open("conll2003_dataset_test.json", "r") as f:
    zero_shot_dataset = json.loads(f.read())

tmp_dataset = {}

for x in zero_shot_dataset:
    if x["entity"] not in tmp_dataset:
        tmp_dataset[x["entity"]] = []
    tmp_dataset[x["entity"]].append(copy.deepcopy(x))

# for x in tmp_dataset:
#     print(x, len(tmp_dataset[x]))
#     with open(f"tmp_{x}", "w") as f:
#         print(json.dumps(tmp_dataset[x], indent=4), file=f)
#     print(tmp_dataset[x])

for x in zero_shot_dataset:
    shots = []
    k = 0
    while True:
        k = random.randint(0, len(tmp_dataset[x["entity"]]) - 1)
        tmp = tmp_dataset[x["entity"]][k]
        if tmp != x:
            shots.append(tmp)
        if len(shots) == FEW_SHOT:
            break
    x["example"] = ""
    for tmp in shots:
        x["example"] += f"\nInput: {tmp['sample_text_input']}\nOutput: {tmp['sample_text_output']}"
    x["full_input"] = f"{x['instruction']} Below are some examples:{x['example']}\n\nInput: {x['sample_text_input']}\nOutput:"

with open("conll2003_dataset_fewshot_test.json", "w") as f:
    print(json.dumps(zero_shot_dataset, indent=4), file=f)

print(zero_shot_dataset[0]["full_input"])

