"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from transformers import AutoTokenizer
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' 表示从检查点恢复
out_dir = 'out-chinese-macbert'  # 检查点目录
start = "根据中华人民共和国法律规定，"  # 修改为有意义的起始文本
num_samples = 10  # 生成的样本数量
max_new_tokens = 500  # 每个样本生成的最大token数
temperature = 0.8  # 1.0 = 无变化, < 1.0 = 更保守, > 1.0 = 更随机
top_k = 200  # 保留概率最高的 top_k 个 token
seed = 1337
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', 'cuda:1' 等
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # 是否使用 PyTorch 2.0 编译功能
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 模型初始化
if init_from == 'resume':
    # 从检查点加载模型
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
encode = lambda s: tokenizer.encode(s, add_special_tokens=False)

def decode(token_ids):
    """解码并移除空格"""
    text = tokenizer.decode(token_ids)
    # 移除字符之间的空格，但保留标点符号前后的空格
    return text.replace(' ', '')

# 编码起始提示文本
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 生成文本
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
