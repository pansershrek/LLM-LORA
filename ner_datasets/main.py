import json
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ncbi_disease")
    parser.add_argument("--ner_len", default=70)
    args = parser.parse_args()
    try:
        try:
            dataset_train = load_dataset(args.dataset)["train"]
            try:
                dataset_test = load_dataset(args.dataset)["test"]
            except:
                dataset_test = load_dataset(args.dataset)["validation"]
        except:
            dataset_train = load_dataset(args.dataset, "supervised")["train"]
            try:
                dataset_test = load_dataset(args.dataset, "supervised")["test"]
            except:
                dataset_test = load_dataset(args.dataset, "supervised")["validation"]
    except:
            dataset_train = load_dataset(args.dataset, "default")["train"]
            try:
                dataset_test = load_dataset(args.dataset, "default")["test"]
            except:
                dataset_test = load_dataset(args.dataset, "default")["validation"]

    info = {
        "Dataset name": args.dataset,
        "Train num samples": len(dataset_train),
        "Test num samples": len(dataset_test),
        "Average train dataset's text len (in words)": 0,
        "Max train dataset's text len (in words)": 0,
        "Min train dataset's text len (in words)": 100000000000000,
        "Average test dataset's text len (in words)": 0,
        "Max test dataset's text len (in words)": 0,
        "Min test dataset's text len (in words)": 100000000000000,
        "NER tags": set(),
    }
    for x in dataset_train:
        info["NER tags"].update(set(x["ner_tags"]))
    info["NER tags"] = list(info["NER tags"])

    for y in info["NER tags"]:
        info[f"Ammount of {y} NER in train dataset (in words)"] = 0
        info[f"Average length of {y} NER in train dataset (in words)"] = 0
        info[f"Min length of {y} NER in train dataset (in words)"] = 100000000000000
        info[f"Max length of {y} NER in train dataset (in words)"] = 0
        info[f"Average length of {y} NER in train dataset (in chars)"] = 0
        info[f"Min length of {y} NER in train dataset (in chars)"] = 100000000000000
        info[f"Max length of {y} NER in train dataset (in chars)"] = 0
        info[f"Ammount of {y} NER in train dataset longer or equal than {args.ner_len}"] = 0


        info[f"Ammount of {y} NER in test dataset (in words)"] = 0
        info[f"Average length of {y} NER in test dataset (in words)"] = 0
        info[f"Min length of {y} NER in test dataset (in words)"] = 100000000000000
        info[f"Max length of {y} NER in test dataset (in words)"] = 0
        info[f"Average length of {y} NER in test dataset (in chars)"] = 0
        info[f"Min length of {y} NER in test dataset (in chars)"] = 100000000000000
        info[f"Max length of {y} NER in test dataset (in chars)"] = 0
        info[f"Ammount of {y} NER in test dataset longer or equal than {args.ner_len}"] = 0

    for x in dataset_train:
        if len(x["ner_tags"]) == 0:
            continue
        if len(x["ner_tags"]) != len(x["tokens"]):
            continue
        info["Average train dataset's text len (in words)"] += len(x["ner_tags"])
        info["Max train dataset's text len (in words)"] = max(info["Max train dataset's text len (in words)"], len(x["ner_tags"]))
        info["Min train dataset's text len (in words)"] = min(info["Min train dataset's text len (in words)"], len(x["ner_tags"]))
        cur_tag = None
        i = 0
        old_i = 0
        n = len(x["ner_tags"])
        while i < n:
            if cur_tag is None:
                cur_tag = x["ner_tags"][i]
            else:
                if cur_tag != x["ner_tags"][i]:
                    ner_len = i - old_i
                    tmp = []
                    for j in range(old_i, i):
                        tmp.append(x["tokens"][j])
                    tmp = " ".join(tmp)
                    ner_len_in_chars = len(tmp)
                    old_i = i

                    info[f"Ammount of {cur_tag} NER in train dataset (in words)"] += 1
                    info[f"Average length of {cur_tag} NER in train dataset (in words)"] += ner_len
                    info[f"Min length of {cur_tag} NER in train dataset (in words)"] = min(info[f"Min length of {cur_tag} NER in train dataset (in words)"], ner_len)
                    info[f"Max length of {cur_tag} NER in train dataset (in words)"] = max(info[f"Max length of {cur_tag} NER in train dataset (in words)"], ner_len)

                    info[f"Average length of {cur_tag} NER in train dataset (in chars)"] += ner_len_in_chars
                    info[f"Min length of {cur_tag} NER in train dataset (in chars)"] = min(info[f"Min length of {cur_tag} NER in train dataset (in chars)"], ner_len_in_chars)
                    info[f"Max length of {cur_tag} NER in train dataset (in chars)"] = max(info[f"Max length of {cur_tag} NER in train dataset (in chars)"], ner_len_in_chars)
                    if ner_len_in_chars >= args.ner_len:
                        info[f"Ammount of {cur_tag} NER in train dataset longer or equal than {args.ner_len}"] += 1
                    cur_tag = x["ner_tags"][i]
            i += 1
        ner_len = i - old_i
        tmp = []
        for j in range(old_i, i):
            tmp.append(x["tokens"][j])
        tmp = " ".join(tmp)
        ner_len_in_chars = len(tmp)
        info[f"Ammount of {cur_tag} NER in train dataset (in words)"] += 1
        info[f"Average length of {cur_tag} NER in train dataset (in words)"] += ner_len
        info[f"Min length of {cur_tag} NER in train dataset (in words)"] = min(info[f"Min length of {cur_tag} NER in train dataset (in words)"], ner_len)
        info[f"Max length of {cur_tag} NER in train dataset (in words)"] = max(info[f"Max length of {cur_tag} NER in train dataset (in words)"], ner_len)
        info[f"Average length of {cur_tag} NER in train dataset (in chars)"] += ner_len_in_chars
        info[f"Min length of {cur_tag} NER in train dataset (in chars)"] = min(info[f"Min length of {cur_tag} NER in train dataset (in chars)"], ner_len_in_chars)
        info[f"Max length of {cur_tag} NER in train dataset (in chars)"] = max(info[f"Max length of {cur_tag} NER in train dataset (in chars)"], ner_len_in_chars)
        if ner_len_in_chars >= args.ner_len:
            info[f"Ammount of {cur_tag} NER in train dataset longer or equal than {args.ner_len}"] += 1

    info["Average train dataset's text len (in words)"] /= info["Train num samples"]

    for x in dataset_test:
        if len(x["ner_tags"]) == 0:
            continue
        if len(x["ner_tags"]) != len(x["tokens"]):
            continue
        info["Average test dataset's text len (in words)"] += len(x["ner_tags"])
        info["Max test dataset's text len (in words)"] = max(info["Max test dataset's text len (in words)"], len(x["ner_tags"]))
        info["Min test dataset's text len (in words)"] = min(info["Min test dataset's text len (in words)"], len(x["ner_tags"]))
        cur_tag = None
        i = 0
        old_i = 0
        n = len(x["ner_tags"])
        while i < n:
            if cur_tag is None:
                cur_tag = x["ner_tags"][i]
            else:
                if cur_tag != x["ner_tags"][i]:
                    ner_len = i - old_i
                    tmp = []
                    for j in range(old_i, i):
                        tmp.append(x["tokens"][j])
                    tmp = " ".join(tmp)
                    ner_len_in_chars = len(tmp)
                    old_i = i

                    info[f"Ammount of {cur_tag} NER in test dataset (in words)"] += 1
                    info[f"Average length of {cur_tag} NER in test dataset (in words)"] += ner_len
                    info[f"Min length of {cur_tag} NER in test dataset (in words)"] = min(info[f"Min length of {cur_tag} NER in test dataset (in words)"], ner_len)
                    info[f"Max length of {cur_tag} NER in test dataset (in words)"] = max(info[f"Max length of {cur_tag} NER in test dataset (in words)"], ner_len)

                    info[f"Average length of {cur_tag} NER in test dataset (in chars)"] += ner_len_in_chars
                    info[f"Min length of {cur_tag} NER in test dataset (in chars)"] = min(info[f"Min length of {cur_tag} NER in test dataset (in chars)"], ner_len_in_chars)
                    info[f"Max length of {cur_tag} NER in test dataset (in chars)"] = max(info[f"Max length of {cur_tag} NER in test dataset (in chars)"], ner_len_in_chars)
                    cur_tag = x["ner_tags"][i]
                    if ner_len_in_chars >= args.ner_len:
                        info[f"Ammount of {cur_tag} NER in test dataset longer or equal than {args.ner_len}"] += 1
            i += 1
        ner_len = i - old_i
        tmp = []
        for j in range(old_i, i):
            tmp.append(x["tokens"][j])
        tmp = " ".join(tmp)
        ner_len_in_chars = len(tmp)
        info[f"Ammount of {cur_tag} NER in test dataset (in words)"] += 1
        info[f"Average length of {cur_tag} NER in test dataset (in words)"] += ner_len
        info[f"Min length of {cur_tag} NER in test dataset (in words)"] = min(info[f"Min length of {cur_tag} NER in test dataset (in words)"], ner_len)
        info[f"Max length of {cur_tag} NER in test dataset (in words)"] = max(info[f"Max length of {cur_tag} NER in test dataset (in words)"], ner_len)
        info[f"Average length of {cur_tag} NER in test dataset (in chars)"] += ner_len_in_chars
        info[f"Min length of {cur_tag} NER in test dataset (in chars)"] = min(info[f"Min length of {cur_tag} NER in test dataset (in chars)"], ner_len_in_chars)
        info[f"Max length of {cur_tag} NER in test dataset (in chars)"] = max(info[f"Max length of {cur_tag} NER in test dataset (in chars)"], ner_len_in_chars)
        if ner_len_in_chars >= args.ner_len:
            info[f"Ammount of {cur_tag} NER in test dataset longer or equal than {args.ner_len}"] += 1

    info["Average test dataset's text len (in words)"] /= info["Test num samples"]

    for y in info["NER tags"]:
        if info[f"Ammount of {y} NER in train dataset (in words)"] > 0:
            info[f"Average length of {y} NER in train dataset (in words)"] /= info[f"Ammount of {y} NER in train dataset (in words)"]
            info[f"Average length of {y} NER in train dataset (in chars)"] /= info[f"Ammount of {y} NER in train dataset (in words)"]

        if info[f"Ammount of {y} NER in test dataset (in words)"] > 0:
                    info[f"Average length of {y} NER in test dataset (in words)"] /= info[f"Ammount of {y} NER in test dataset (in words)"]
                    info[f"Average length of {y} NER in test dataset (in chars)"] /= info[f"Ammount of {y} NER in test dataset (in words)"]

    with open(f"{args.dataset}_info.json".replace("/", "_"), "w") as f:
        print(json.dumps(info, indent=4), file=f)


if __name__ == "__main__":
    main()
