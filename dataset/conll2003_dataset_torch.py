import json

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class Conll2003Dataset(Dataset):
    def __init__(self, split, tokenizer, max_length=1102):
        self.instruction = (
            "I am an excellent linguist. "
            "The task is to label {entity} entities in the given sentence."
        )
        self.entity_types = ['PER', 'ORG', 'LOC', 'MISC']
        self.entity_types_full = {
            'PER': 'PER (persons)',
            'ORG': 'ORG (organizations)',
            'LOC': 'LOC (locations)',
            'MISC': 'MISC (miscellaneous names)'
        }
        self.tagset = {
            'O': 0, 'B-PER': 1, 'I-PER': 2,
            'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5,
            'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8
        }
        self.tagset_rev = dict(zip(self.tagset.values(), self.tagset.keys()))
        self.begin_entity_symbol = "@@"
        self.end_entity_symbol = "##"
        self.samples, self.max_len_sample_text = (
            self.creat_samples(load_dataset('conll2003', split=split))
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def fix_sample_text(self, text):
        output = text.replace("@@ ", "@@").replace(' ##', '##')
        #output = text.replace(" .", ".").replace(" ,", ",")
        #output = output.replace(" !", "!").replace(" ?", "?")
        #output = output.replace(" :", ":").replace(" -", "-")
        #output = output.replace(" '", "'").replace(' "', '"')
        #output = output.replace("@@ ", "@@").replace(' ##', '##')
        #output = output.replace(" =", "=").replace(" /", "/")
        #output = output.replace("- ", "-")
        return output

    def convert_ner_tags(self, tags):
        ans = []
        for x in tags:
            tag = self.tagset_rev[x]
            if "-" in tag:
                tag = tag[2:]
            ans.append(tag)
        return ans

    def convert_sample(self, ner_tags, tokens, cur_ner_tag):
        begin_indexes = []
        end_indexes = []
        prev_tag = None
        terminal_tagset = {}
        for k, v in self.tagset.items():
            if not k.endswith(cur_ner_tag):
                terminal_tagset[v] = k

        for idx, (tag, token) in enumerate(zip(ner_tags, tokens)):
            tag_label = (
                list(self.tagset.keys())[list(self.tagset.values()).index(tag)]
            )
            if (
                tag in terminal_tagset and prev_tag is not None
                and prev_tag.endswith(cur_ner_tag)
            ):
                end_indexes.append(idx - 1)
            else:
                if tag_label == f"B-{cur_ner_tag}":
                    if (
                        prev_tag == f"I-{cur_ner_tag}" or
                        prev_tag == f"B-{cur_ner_tag}"
                    ):
                        end_indexes.append(idx - 1)
                    begin_indexes.append(idx)

            prev_tag = tag_label
        if prev_tag.endswith(cur_ner_tag):
            end_indexes.append(len(ner_tags) - 1)


        if len(begin_indexes) != len(end_indexes):
            raise Exception("len(begin_indexes) != len(end_indexes)")

        sample = []
        begin_indexes = set(begin_indexes)
        end_indexes = set(end_indexes)
        for idx, token in enumerate(tokens):
            if idx in begin_indexes:
                sample.append(self.begin_entity_symbol)
            sample.append(token)
            if idx in end_indexes:
                sample.append(self.end_entity_symbol)

        return self.fix_sample_text(" ".join(sample))

    def creat_samples(self, dataset):
        samples = []
        max_len_sample_text = 0
        for x in dataset:
            for entity_idx, entity in enumerate(self.entity_types):
                try:
                    tmp = {
                        "entity": entity,
                        "text_id": x["id"],
                        "full_id": f"{x['id']}_{entity_idx}",
                        "sample_text_output": self.convert_sample(
                            x["ner_tags"], x["tokens"], entity
                        ),
                        "sample_text_input": self.fix_sample_text(
                            " ".join(x["tokens"])
                        ),
                        #"original_ner_tags": x["ner_tags"],
                        #"original_tokens": x["tokens"],
                        "converted_ner_tags": (
                            self.convert_ner_tags(x["ner_tags"])
                        ),
                    }
                    tmp["instruction"] = (
                        self.instruction.format(
                            entity=self.entity_types_full[tmp["entity"]]
                        )
                    )
                    tmp["input"] = (
                        f'{tmp["instruction"]}\n'
                        f"Input: {tmp['sample_text_input']}\n"
                        f"Output:"
                    )
                    tmp["output"] = f" {tmp['sample_text_output']}"
                    tmp["full_input"] = (
                        f"{tmp['input']}{tmp['output']}"
                    )
                    samples.append(tmp)
                except:
                    raise

                max_len_sample_text = max(
                    max_len_sample_text,
                    len(samples[-1]["full_input"])
                )
        return samples, max_len_sample_text

    def get_input_max_len(self):
        if self.max_len_sample_text > self.max_length:
            return self.max_length
        return self.max_len_sample_text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tmp = self.samples[index]
        return tmp
        # input_ids = self.tokenizer(
        #     tmp["full_input"],
        #     truncation=True,
        #     max_length=self.get_input_max_len(),
        #     padding="max_length"
        # )["input_ids"]
        # input_ids_not_mask = self.tokenizer(
        #     tmp["output"],
        #     truncation=True
        # )["input_ids"]
        # mask_len = (
        #     len(input_ids) - len(input_ids_not_mask)
        # )
        # return {
        #     "input_ids": input_ids,
        #     "labels": (
        #         [-100] * mask_len + input_ids[mask_len:]
        #     ),
        #     "attention_mask": [1] * len(input_ids),
        # }

ds = Conll2003Dataset("train", None)
tmp = []

for x in ds:
    tmp.append(x)

with open("conll2003_dataset_train.json", "w") as f:
    print(json.dumps(tmp, indent=4), file=f)

print(ds.max_len_sample_text)

ds = Conll2003Dataset("test", None)
tmp = []

for x in ds:
    tmp.append(x)

with open("conll2003_dataset_test.json", "w") as f:
    print(json.dumps(tmp, indent=4), file=f)

print(ds.max_len_sample_text)

