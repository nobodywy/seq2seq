'''
seq2seq基于词汇级,单项LSTM网络
'''
from keras.models import Model
import numpy as np
import pickle
import jieba
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
import os
from keras.models import load_model
from keras.layers import  Concatenate, Dot, Input, LSTM, Embedding, Dense
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import random
import math
import json

# 设置超参数
New_model = True #是否初次训练
loaded_model_path = './model/attention_model_v2.h5'  #model path
new_model_path =  './model/attention_model_v2' #.h5'


# 设置超参数
batch_size = 64  # 批次大小
epochs = 50  # 迭代次数
latent_dim = 256  # LSTM隐藏单元的数量
num_samples = 15000  # 训练样本大小
input_word_num = 2000  # 设置输入词汇的大小
target_word_num = 4000  # 设置输出词汇的大小
max_encoder_seq_length = 135  # 输入句子的最大词汇长度
max_decoder_seq_length = 220  # 输出句子的最大词汇长度
embedding_dim = 300  # embedding层维数
dense1_shape = 8
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

# # 保存或加载字符集
# if not os.path.exists('./data/word_count.pkl'):
#     with open('./data/word_count.pkl', 'wb') as f:
#         pickle.dump(input_words_count, f)
#         pickle.dump(target_words_count, f)
# else:
#     with open('./data/word_count.pkl', 'rb') as f:
#         input_words_count = pickle.load(f)
#         target_words_count = pickle.load(f)

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
        if t > 0:
            try:
                decoder_target_data[i, t - 1] = target_token_index[word]
            except:
                decoder_target_data[i, t - 1] = target_token_index['<UNK>']

# 将decoder_target_data转化为one-hot形式
decoder_target_data = to_categorical(decoder_target_data)

# 定义softmax函数
def softmax(x):
    return K.softmax(x, axis=1)

# 定义相关层
at_repeat = RepeatVector(max_encoder_seq_length)
at_concatenate = Concatenate(axis=-1)
at_dense1 = Dense(dense1_shape, activation="tanh")
at_dense2 = Dense(1, activation="relu")
at_softmax = Activation(softmax, name='attention_weights')
at_dot = Dot(axes=1)
layer3 = Dense(num_decoder_tokens, activation=softmax)


# 定义单步注意力机制层，计算单步的上下文向量
def one_step_of_attention(h_prev, a):
    """
    计算上下文向量

    Input:
    h_prev：decoder层前一个时间步的隐藏状态
    a：encoder层的隐藏层状态(m, max_encoder_seq_length , n_a)

    Output:
    context：上下文向量(m,1,n_a)
    """
    # 将前一个时间步的隐藏层状态复制Tx次，变成(m,max_encoder_seq_length,n_h)的格式
    h_repeat = at_repeat(h_prev)
    # 与a进行连接，输出（m,max_encoder_seq_length,n_a+n_h）
    i = at_concatenate([a, h_repeat])
    # 全连接层1，输出（m,max_encoder_seq_length,dense1_shape）
    i = at_dense1(i)
    # 全连接层2，输出（m,max_encoder_seq_length,1）
    i = at_dense2(i)
    # softmax层
    attention = at_softmax(i)
    # 计算上下文向量
    context = at_dot([attention, a])

    return context


# 注意力机制层
def attention_layer(X, latent_dim, max_decoder_seq_length):
    """
    注意力机制层

    Input:
    X：encoder层的输出(m, Tx, n_a)
    latent_dim：decoder层隐藏单元的数量
    max_decoder_seq_length：输出句子的最大长度

    Output:
    output - The output of the attention layer (max_decoder_seq_length,m,n_h)
    """
    # 用0初始化h、c
    h = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], latent_dim)))(X)
    c = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], latent_dim)))(X)

    at_LSTM = LSTM(latent_dim, return_state=True)
    output = []

    # Run attention step and RNN for each output time step
    for _ in range(max_decoder_seq_length):
        context = one_step_of_attention(h, X)
        h, _, c = at_LSTM(context, initial_state=[h, c])
        output.append(h)

    return output


# 建立模型
def build_basic_model():
    # Encoder
    encoder_inputs = Input(shape=(max_encoder_seq_length,), name='encoder_inputs')
    encoder_embedding = Embedding(num_encoder_tokens, embedding_dim, name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
    encoder_outputs, *encoder_states = encoder_lstm(encoder_embedding)

    # Attention
    attention_outputs = attention_layer(encoder_outputs, latent_dim, max_decoder_seq_length)
    decoder_outputs = [layer3(timestep) for timestep in attention_outputs]
    decoder_outputs =
    # 创建模型
    basic_model = Model(inputs=[encoder_inputs], outputs=decoder_outputs)
    basic_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return basic_model



# 训练模型
#basic_model = build_basic_model()
#decoder_target_data = list(decoder_target_data.swapaxes(0, 1))  # 交换前两维


# 获取模型
if(New_model):
    basic_model = build_basic_model()
else:
    basic_model = load_model(loaded_model_path)

# 训练
epochs = 10
for i in range(0,5):
    basic_model_hist = basic_model.fit(encoder_input_data, decoder_target_data, batch_size=batch_size, epochs=epochs)
    # 保存模型
    basic_model.save(new_model_path+'_epo_'+str(int(i)*10+epochs)+'.h5')

#basic_model.fit([encoder_input_data], decoder_target_data, epochs=epochs, batch_size=batch_size)

# # 保存模型
# basic_model.save('./model/word_model_base_with_att.h5')
#
# # 字符逆字典
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
#
#
# # 绘制模型结构
# plot_model(basic_model, to_file='./model/word_model_base_with_att.png', show_shapes=True, show_layer_names=True)
#
# # 模型预测
pred=basic_model.predict(encoder_input_data[0:1])
max_prediction = [y.argmax() for y in pred]
str_prediction = [reverse_target_char_index[y] for y in max_prediction if y!=0]


