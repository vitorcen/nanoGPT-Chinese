out_dir = 'out-chinese-poetry'
eval_interval = 50 # save and evaluate every 50 iterations
eval_iters = 10 # evaluate every 10 iterations
log_interval = 2 # log every 2 iterations

always_save_checkpoint = True # save checkpoint every iteration

wandb_log = True
wandb_project = 'chinese-poetry'
wandb_run_name = 'gpt-chinese-poetry'

dataset = 'chinese_modern_poetry' # dataset name

gradient_accumulation_steps = 8 # 多少步更新一次参数，越小训练速度越快：4 - 约 1 iter 半秒, 8 - 约 1 iter 1秒
batch_size = 32 # 实际每次处理的样本数，越大需要越多显存：32 - 需要18G显存，16需要12G显存
#总样本数 = gradient_accumulation_steps * batch_size

block_size = 1024 # sequence length
# 每个迭代的 token 数=batch size×block size

n_layer = 12 # number of layers
n_head = 12 # number of attention heads
n_embd = 768 # embedding dimension
dropout = 0.01 # dropout rate

#about 4s per iter, (300 * 4000ms) / (1000 * 60 * 60) ≈ 20 minutes
learning_rate = 6e-4 # max learning rate
# 300 * 19456 = 5,836,800 tokens
max_iters = 150 #train.bin tokens will be read about 4.6 times
lr_decay_iters = 150
min_lr = 6e-5 # minimum learning rate
beta2 = 0.95 # beta2 for AdamW optimizer

warmup_iters = 30 # how many steps to warm up for

# device and compilation
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster