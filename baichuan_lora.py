import warnings

warnings.filterwarnings('ignore')
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from mydatasets import MyDataset
from my_data_collator import data_collator
from peft import get_peft_model, TaskType
from peft import LoraConfig
from torchkeras import KerasModel
from models import StepRunner
from torch.optim.lr_scheduler import CosineAnnealingLR

model_name_or_path = 'baichuan-13b'  # 联网远程加载 'baichuan-inc/Baichuan-13B-Chat'

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
messages = []
messages.append({"role": "user",
                 "content": "世界上第二高的山峰是哪座?"})
response = model.chat(tokenizer, messages=messages, stream=True)
for res in response:
    print(res, end='\r')
filename = r'/Users/haojingkun/PycharmProjects/llama_tuning/data/bankuai.json'
conversation = []
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        json_data = json.loads(line.strip())
        history = json_data['history']
        instruction = json_data['instruction']
        output = json_data['output']
        instructions = []
        outputs = []
        for his in history:
            instructions.append(his[0])
            outputs.append(his[1])
        instructions.append(instruction)
        outputs.append(output)
        conversation.append((instructions, outputs))

ds_train = ds_val = MyDataset(conversation, model, tokenizer)
dl_train = torch.utils.data.DataLoader(ds_train, num_workers=2, batch_size=4,
                                       pin_memory=True, shuffle=True,
                                       collate_fn=data_collator)
dl_val = torch.utils.data.DataLoader(ds_val, num_workers=2, batch_size=4,
                                     pin_memory=True, shuffle=False,
                                     collate_fn=data_collator)

model.supports_gradient_checkpointing = True  #
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['up_proj', 'down_proj', 'o_proj', 'gate_proj', 'W_pack']
)

peft_model = get_peft_model(model, peft_config)

peft_model.is_parallelizable = True
peft_model.model_parallel = True
peft_model.print_trainable_parameters()
KerasModel.StepRunner = StepRunner


# 仅仅保存QLoRA的可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator=None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)


def load_ckpt(self, ckpt_path='checkpoint'):
    self.net = self.net.from_pretrained(self.net.base_model.model,
                                        ckpt_path, is_trainable=True)
    self.from_scratch = False


KerasModel.save_ckpt = save_ckpt
KerasModel.load_ckpt = load_ckpt
lr_scheduler = CosineAnnealingLR(torch.optim.AdamW(model.parameters(), lr=5e-4), T_max=10)
keras_model = KerasModel(model, loss_fn=None,
                         optimizer=torch.optim.AdamW(model.parameters(), lr=5e-4), lr_scheduler=lr_scheduler)
ckpt_path = 'baichuan13b_multi_rounds'
keras_model.fit(train_data=dl_train,
                val_data=dl_val,
                epochs=100, patience=10,
                monitor='val_loss', mode='min',
                ckpt_path=ckpt_path
                )
