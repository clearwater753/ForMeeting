from transformers import BertTokenizer, BertModel

# 初始化分词器和模型
path = r'D:\Desktop\lee_code-hot100\自然语言处理\bert-bae-uncased'
tokenizer = BertTokenizer.from_pretrained(path)
model = BertModel.from_pretrained(path)

# 输入文本
text = "Hello, how are you?"

# # 分词
# tokens = tokenizer.tokenize(text)
# print("分词结果：", tokens)

# # 转换为Token ID
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print("Token ID：", token_ids)

# 获取Token Embedding
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]  # 合并了分词和映射id操作
embeddings = model.embeddings.word_embeddings(input_ids)  # 获取Token Embedding
print("Token Embedding：", embeddings.shape)  # 输出嵌入向量的形状