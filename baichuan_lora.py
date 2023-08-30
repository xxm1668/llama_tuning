import warnings

warnings.filterwarnings('ignore')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from peft import get_peft_model, TaskType
from peft import LoraConfig
from torchkeras import KerasModel
from models import StepRunner
from torch.optim.lr_scheduler import CosineAnnealingLR
from my_data_collator import split_data, get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = '/home/xxm/model/internlm-chat-7b'  # 联网远程加载 'baichuan-inc/Baichuan-13B-Chat'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

model.half().cuda()
model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
messages = []
messages.append({"role": "user",
                 "content": "世界上第二高的山峰是哪座?"})
response = model.chat(tokenizer, query=messages[0]['content'])
for res in response:
    print(res, end='\r')
filename = r'/home/xxm/fsdownload/llama_tuning/data/estate_qa11_.json'
ds_train, ds_val = split_data(filename)
ds_train, ds_val = get_data(ds_train, ds_val)
for train in ds_train:
    train['input_ids'].to(device)
    train['labels'].to(device)
for val in ds_val:
    val['input_ids'].to(device)
    val['labels'].to(device)

model.supports_gradient_checkpointing = True  #
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=32,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj']
)

peft_model = get_peft_model(model, peft_config)

peft_model.is_parallelizable = True
peft_model.model_parallel = True
peft_model.print_trainable_parameters()
KerasModel.StepRunner = StepRunner

KerasModel.save_ckpt = StepRunner.save_ckpt
KerasModel.load_ckpt = StepRunner.load_ckpt
lr_scheduler = CosineAnnealingLR(torch.optim.AdamW(model.parameters(), lr=5e-4), T_max=10)
keras_model = KerasModel(model, loss_fn=None,
                         optimizer=torch.optim.AdamW(model.parameters(), lr=5e-4), lr_scheduler=lr_scheduler)
ckpt_path = 'baichuan13b_multi_rounds'
keras_model.fit(train_data=ds_train,
                val_data=ds_val,
                epochs=50, patience=100,
                monitor='val_loss', mode='min',
                ckpt_path=ckpt_path
                )
