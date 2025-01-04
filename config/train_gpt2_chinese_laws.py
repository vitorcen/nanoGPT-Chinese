# config for training with hfl/chinese-macbert-base

out_dir = 'out-chinese-laws'
eval_interval = 1000 # save and evaluate every 1000 iterations
eval_iters = 200 # evaluate every 200 iterations
log_interval = 10 # log every 10 iterations

always_save_checkpoint = True # save checkpoint every iteration

wandb_log = True
wandb_project = 'chinese-laws'
wandb_run_name = 'gpt-chinese-laws'

dataset = 'chinese_laws_pretrain' # dataset name
gradient_accumulation_steps = 5 * 8 # simulate larger batch sizes
# 每个迭代的 token 数=batch size×block size=19×1024=19456 tonens
batch_size = 19 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024 # sequence length

n_layer = 12 # number of layers
n_head = 12 # number of attention heads
n_embd = 768 # embedding dimension
dropout = 0.01 # dropout rate

#about 4s per iter, (5000 * 4000ms) / (1000 * 60 * 60) ≈ 5.5 hours
learning_rate = 6e-4 # max learning rate
# 5000 * 19456 = 97280000 tokens
max_iters = 5000 #50000, train.bin 1,278,441 tokens read about 76 times
lr_decay_iters = 5000 #50000
min_lr = 6e-5 # minimum learning rate
beta2 = 0.95 # beta2 for AdamW optimizer

warmup_iters = 2000 # how many steps to warm up for

# device and compilation
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster