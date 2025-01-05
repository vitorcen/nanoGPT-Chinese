from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 使用 RoBERTa 的中英双语词表
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

text = "GPT-4 是一个强大的语言模型，能够处理多种语言。它的应用包括：翻译、问答、生成文本等。"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("IDs:", ids)
