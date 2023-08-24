from torch.utils.data import Dataset, DataLoader
import random


# 准备数据
def get_messages(conversation):
    select = random.choice
    messages, history = [], []
    for t in conversation:
        history.append((select(t[0]), select(t[-1])))

    for prompt, response in history:
        pair = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        messages.extend(pair)
    return messages


# reference@ model._build_chat_input?
def build_chat_input(messages, model, tokenizer, max_new_tokens=None):
    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    max_input_tokens = max(model.config.model_max_length // 2, max_input_tokens)

    total_input, round_input, total_label, round_label = [], [], [], []

    for i, message in enumerate(messages[::-1]):
        content_tokens = tokenizer.encode(message['content'])
        if message['role'] == 'user':
            round_input = [model.generation_config.user_token_id] + content_tokens + round_input
            round_label = [-100] + [-100 for _ in content_tokens] + round_label

            if total_input and len(total_input) + len(round_input) > max_input_tokens:
                break
            else:
                total_input = round_input + total_input
                total_label = round_label + total_label
                if len(total_input) >= max_input_tokens:
                    break
                else:
                    round_input = []
                    round_label = []

        elif message['role'] == 'assistant':
            round_input = [
                              model.generation_config.assistant_token_id
                          ] + content_tokens + [
                              model.generation_config.eos_token_id
                          ] + round_input

            round_label = [
                              -100
                          ] + content_tokens + [
                              model.generation_config.eos_token_id  # 注意，除了要学习机器人回复内容，还要学习一个结束符。
                          ] + round_label
        else:
            raise ValueError(f"message role not supported yet: {message['role']}")

    total_input = total_input[-max_input_tokens:]  # truncate left
    total_label = total_label[-max_input_tokens:]

    total_input.append(model.generation_config.assistant_token_id)
    total_label.append(-100)

    return total_input, total_label


class MyDataset(Dataset):
    def __init__(self, conv, model, tokenizer, size=8
                 ):
        super().__init__()
        self.conv = conv
        self.model = model
        self.tokenizer = tokenizer
        self.__dict__.update(locals())

    def __len__(self):
        return self.size

    def get(self, index):
        messages = get_messages(self.conv)
        return messages

    def __getitem__(self, index):
        messages = self.get(index)
        input_ids, labels = build_chat_input(messages, self.model, self.tokenizer)
        return {'input_ids': input_ids, 'labels': labels}
