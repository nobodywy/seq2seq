#coding=utf-8
'''
seq2seq基于词汇级
'''

import numpy as np
import jieba
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import os
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils import plot_model
#from models.custom_recurrents import AttentionDecoder
from keras.layers.recurrent import Recurrent
import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from keras.utils import plot_model
from models.custom_recurrents import AttentionDecoder






# 设置超参数
New_model = True #是否初次训练
model_path = './model/word_model.h5'  #model path
new_model_path =  './model/word_model_100ep.h5'

batch_size = 64  # 批次大小
epochs = 50  # 迭代次数
latent_dim = 256  # LSTM隐藏单元的数量
num_samples = 15000  # 训练样本大小
input_word_num = 2000  # 设置输入词汇的大小
target_word_num = 4000  # 设置输出词汇的大小
max_encoder_seq_length = 220  # 输入句子的最大词汇长度
max_decoder_seq_length = 220  # 输出句子的最大词汇长度
embedding_dim = 300  # embedding层维数
data_path = './data/ans_com.txt'  # 数据路径

# 读取数据，并获取输入数据和输出数据的字符集
input_texts = []  # 输入字符串列表
target_texts = []  # 输出字符串列表
input_words_count = dict()
target_words_count = dict()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # 对输出序列前后添加'\t'和'\n'标记
    target_text = '\t' + target_text + '\n'
    # 对文本进行分词
    input_text = list(jieba.cut(input_text))
    target_text = list(jieba.cut(target_text))
    input_texts.append(' '.join(input_text))
    target_texts.append(' '.join(target_text))

# 获取前word_num个词汇
tokenizer_input = Tokenizer(num_words=input_word_num, filters='')
tokenizer_input.fit_on_texts(input_texts)
input_words_count = tokenizer_input.word_counts

tokenizer_output = Tokenizer(num_words=target_word_num, filters='')
tokenizer_output.fit_on_texts(target_texts)
target_words_count = tokenizer_output.word_counts

# 对词汇出现的次数进行排序
input_words_count = sorted(input_words_count.items(), key=lambda x: x[1], reverse=True)
target_words_count = sorted(target_words_count.items(), key=lambda x: x[1], reverse=True)

# 选取前word_num个词汇，其他字符集统一用未知词汇符号，即UNK表示
input_words_count_select = input_words_count[:input_word_num]
target_words_count_select = target_words_count[:target_word_num]

# 对词汇进行排序
input_words = sorted([i[0] for i in input_words_count_select])
target_words = sorted([i[0] for i in target_words_count_select])

# 增加未知字符
input_words.append('<UNK>')
target_words.append('<UNK>')

# 增加pad字符
input_words = ['<PAD>'] + input_words
target_words = ['<PAD>'] + target_words

# 获取输入词汇集和输出词汇集的长度
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)

# 构建词汇集字典
input_token_index = dict([(word, i) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i) for i, word in enumerate(target_words)])

# 将输入和输出数据都转化为整数序列格式
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype='int')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype='int')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    input_text = input_text.split(' ')
    target_text = target_text.split(' ')
    if len(target_text) > max_decoder_seq_length:
        target_text = target_text[:(max_decoder_seq_length - 1)]
        target_text.append('\n')
    for t, word in enumerate(input_text[:max_encoder_seq_length]):
        try:
            encoder_input_data[i, t] = input_token_index[word]
        except:
            encoder_input_data[i, t] = input_token_index['<UNK>']
    for t, word in enumerate(target_text[:max_decoder_seq_length]):
        # decoder_target_data要比decoder_input_data早一个时间步
        try:
            decoder_input_data[i, t] = target_token_index[word]
        except:
            decoder_input_data[i, t] = target_token_index['<UNK>']
        if t > 0:
            try:
                decoder_target_data[i, t - 1] = target_token_index[word]
            except:
                decoder_target_data[i, t - 1] = target_token_index['<UNK>']

# 将decoder_target_data转化为one-hot形式
#decoder_target_data = to_categorical(decoder_target_data)

mod = load_model('./model/attention_model_gpu_epo_200.h5',custom_objects={'AttentionDecoder': AttentionDecoder})
res = mod.predict(encoder_input_data[1:2])
# 字符逆字典
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

word1 = res[0]
res_context = []
for word in res[:5]:
    index = np.argmax(word)
    res_context.append(reverse_target_char_index[index])
print("".join(res_context))

