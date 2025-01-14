import os
import json
from transformers import AutoTokenizer
import numpy as np
import pickle
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
import shutil

# [
# {
#     "title": "中华人民共和国消防法",
#     "content": "第六十六条 电器产品、燃气用具的安装、使用及其线路、管路的设计、敷设、维护保养、检测不符合消防技术标准和管理规定的，责令限期改正；逾期不改正的，责令停止使用，可以并处一千元以上五千元以下罚款。\n"
# },
# {
#     "title": "中华人民共和国非物质文化遗产法",
#     "content": "第三十五条 图书馆、文化馆、博物馆、科技馆等公共文化机构和非物质文化遗产学术研究机构、保护机构以及利用财政性资金举办的文艺表演团体、演出场所经营单位等，应当根据各自业务范围，开展非物质文化遗产的整理、研究、学术交流和非物质文化遗产代表性项目的宣传、展示。\n"
# },
# ]

def get_dataset_path(dataset_name="Dusker/chinese-laws-pretrain"):
    """获取数据集路径，如果本地不存在则下载"""
    try:
        # 尝试使用 snapshot_download 下载整个数据集
        dataset_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=None,  # 使用默认的缓存目录
            ignore_patterns=[".*"],
        )
        print(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

# 获取数据集路径
dataset_dir = get_dataset_path()
if dataset_dir is None:
    raise ValueError("Failed to download or locate the dataset")

# 初始化训练和验证数据存储
train_data = []
val_data = []

# 遍历目录中的所有 JSON 文件
print("\nProcessing JSON files:")
for filename in os.listdir(dataset_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(dataset_dir, filename)
        print(f"Reading file: {filename}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            entries_count = len(data)
            
            # 计算分割点
            split_idx = int(len(data) * 0.9)
            
            # 分割数据
            train_entries = data[:split_idx]
            val_entries = data[split_idx:]
            
            # 将title和content组合在一起
            train_data.extend([f"{entry['title']}{entry['content']}" for entry in train_entries])
            val_data.extend([f"{entry['title']}{entry['content']}" for entry in val_entries])
            
            print(f"  - Processed {entries_count} entries from {filename}")
            print(f"    Train: {len(train_entries)}, Val: {len(val_entries)}")

print(f"\nTotal files processed: {len([f for f in os.listdir(dataset_dir) if f.endswith('.json')])}")
print(f"Total entries: Train = {len(train_data)}, Val = {len(val_data)}\n")

# 合并训练和验证数据
train_text = ' '.join(train_data)
val_text = ' '.join(val_data)

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

# 使用分词器进行分词
train_tokens = tokenizer.tokenize(train_text)
val_tokens = tokenizer.tokenize(val_text)

# 将分词结果转换为整数 ID
train_ids = tokenizer.convert_tokens_to_ids(train_tokens)
val_ids = tokenizer.convert_tokens_to_ids(val_tokens)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 创建输出目录
output_dir = os.path.dirname(__file__)

# 保存 meta.pkl 包含 vocab_size
meta = {
    'vocab_size': tokenizer.vocab_size,
    'other_info': '任何其他您需要保存的信息'
}

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 保存 meta.pkl 到输出目录
meta_path = os.path.join(output_dir, 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

# 导出到 bin 文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

print(f"Saved meta.pkl, train.bin, and val.bin to {output_dir}")
print(f"train.bin has {len(train_ids):,} tokens")
print(f"val.bin has {len(val_ids):,} tokens")

# train.bin has 1,394,452 tokens
# val.bin has 170,524 tokens
