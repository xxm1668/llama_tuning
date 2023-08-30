import torch
from transformers import AutoTokenizer, LlamaConfig
import datasets
from sklearn.model_selection import train_test_split
import json

model_name_or_path = '/home/xxm/model/internlm-chat-7b'  # 联网远程加载 'baichuan-inc/Baichuan-13B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
config = LlamaConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
max_seq_length = 512
skip_over_length = True


def data_collator(features: list):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

    input_ids = []
    labels_list = []

    for length, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        context_len = feature["context_len"]

        labels = (
                [-100] * (context_len - 1) + ids[(context_len - 1):] + [-100] * (longest - length)
        )  # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss

        ids = ids + [tokenizer.pad_token_id] * (longest - length)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labels))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# 将上下文整理成与推理时候一致，参照model.chat中的源码~
# model.build_inputs??
template = r'<|User|>:{query}<eoh>\n<|Bot|>:{response}<eoa>\n'


def build_inputs(query, history):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "<|User|>:{}<eoh>\n<|Bot|>:{}<eoa>\n".format(old_query, response)
    prompt += "<|User|>:{}<eoh>\n<|Bot|>:".format(query)
    return prompt


def split_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data.append(json.loads(line))
    dftrain, dftest = train_test_split(data, test_size=0.01, random_state=42)
    train = []
    for x in dftrain:
        tmp = {}
        tmp['context'] = build_inputs(x['instruction'], history=x['history'])
        tmp['target'] = x['output']
        train.append(tmp)

    test = []
    for x in dftest:
        tmp = {}
        tmp['context'] = build_inputs(x['instruction'], history=x['history'])
        tmp['target'] = x['output']
        test.append(tmp)

    ds_train = datasets.Dataset.from_list(train)
    ds_val = datasets.Dataset.from_list(test)

    return ds_train, ds_val


def preprocess(example):
    context = example["context"]
    target = example["target"]

    context_ids = tokenizer.encode(
        context,
        max_length=max_seq_length,
        truncation=True)

    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)

    input_ids = context_ids + target_ids + [config.eos_token_id]

    return {"input_ids": input_ids, "context_len": len(context_ids), 'target_len': len(target_ids)}


def get_data(ds_train, ds_val):
    ds_train_token = ds_train.map(preprocess).select_columns(['input_ids', 'context_len', 'target_len'])
    if skip_over_length:
        ds_train_token = ds_train_token.filter(
            lambda example: example["context_len"] < max_seq_length and example["target_len"] < max_seq_length)

    ds_val_token = ds_val.map(preprocess).select_columns(['input_ids', 'context_len', 'target_len'])
    if skip_over_length:
        ds_val_token = ds_val_token.filter(
            lambda example: example["context_len"] < max_seq_length and example["target_len"] < max_seq_length)
    dl_train = torch.utils.data.DataLoader(ds_train_token, num_workers=2, batch_size=2,
                                           pin_memory=True, shuffle=True,
                                           collate_fn=data_collator)
    dl_val = torch.utils.data.DataLoader(ds_val_token, num_workers=2, batch_size=2,
                                         pin_memory=True, shuffle=True,
                                         collate_fn=data_collator)
    return dl_train, dl_val


if __name__ == '__main__':
    split_data('/home/xxm/fsdownload/llama_tuning/data/estate_qa11_.json')
