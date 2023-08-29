import torch
from transformers import AutoTokenizer

model_name_or_path = 'baichuan-13b'  # 联网远程加载 'baichuan-inc/Baichuan-13B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def data_collator(examples: list):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

    input_ids = []
    labels_list = []

    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]

        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        labs = labs + [-100] * (longest - length)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }
