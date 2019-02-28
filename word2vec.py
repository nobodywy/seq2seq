import pickle
import jieba
import tqdm
from gensim.models.word2vec import Word2Vec
import gensim

# 设置超参数
num_samples = 15000  # 设置数据集大小
data_path = './data/ans_com.txt'  # 数据路径
embedding_size = 100  # 词向量大小
min_count = 0

# 读取数据，并获取输入数据和输出数据的字符集
input_texts = []  # 输入字符串列表
target_texts = []  # 输出字符串列表
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in tqdm.tqdm(lines[: min(num_samples, len(lines) - 1)]):
    input_text, target_text = line.split('\t')
    # 对输出序列前后添加'\t'和'\n'标记
    target_text = target_text + '\n'
    # 对文本进行分词
    input_text = list(jieba.cut(input_text))
    target_text = list(jieba.cut(target_text))
    input_texts.append(input_text)
    target_texts.append(target_text)

# 用word2vec预训练词向量
input_text_w2c = Word2Vec(input_texts, size=embedding_size, min_count=min_count)
target_text_w2c = Word2Vec(target_texts, size=embedding_size, min_count=min_count)

# 保存预训练好的词向量
# with open('./data/input_text_w2c.pkl','wb') as f:
#     pickle.dump(input_text_w2c,f)
#
# with open('./data/target_text_w2c.pkl','wb') as f:
#     pickle.dump(target_text_w2c,f)

input_text_w2c.save('./data/input_text_w2c.model')