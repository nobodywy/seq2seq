'''
seq2seq基于字符集
'''
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras
import os

# 设置超参数
batch_size = 32  # 批次大小
epochs = 5  # 迭代次数
latent_dim = 256  # LSTM隐藏单元的数量
num_samples = 10000  # 训练样本大小
input_char_num = 600  # 设置字符集的大小
target_char_num = 600
data_path = './data/ans_com.txt'  # 数据路径

# 读取数据，并获取输入数据和输出数据的字符集
input_texts = []  # 输入字符串列表
target_texts = []  # 输出字符串列表
input_characters = set()  # 输入字符集
target_characters = set()  # 输出字符集
input_characters_count = dict()
target_characters_count = dict()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
print('finish:load_data')
if not os.path.exists('./data/char_count.pkl'):
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # 对输出序列前后添加'\t'和'\n'标记
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
            if np.isin(char, list(input_characters_count.keys())):
                input_characters_count[char] += 1
            else:
                input_characters_count[char] = 1
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
            if np.isin(char, list(target_characters_count.keys())):
                target_characters_count[char] += 1
            else:
                target_characters_count[char] = 1
else:
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # 对输出序列前后添加'\t'和'\n'标记
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

# 保存或加载字符集
if not os.path.exists('./data/char_count.pkl'):
    with open('./data/char_count.pkl', 'wb') as f:
        pickle.dump(input_characters_count, f)
        pickle.dump(target_characters_count, f)
else:
    with open('./data/char_count.pkl', 'rb') as f:
        input_characters_count = pickle.load(f)
        target_characters_count = pickle.load(f)

# 对字符集出现的次数进行排序
input_characters_count = sorted(input_characters_count.items(), key=lambda x: x[1], reverse=True)
target_characters_count = sorted(target_characters_count.items(), key=lambda x: x[1], reverse=True)


# 选取前char_num个字符集，其他字符集统一用未知字符集符号，即UNK表示
input_characters_count_select = input_characters_count[:input_char_num]
target_characters_count_select = target_characters_count[:target_char_num]

# 对字符集进行排序
input_characters = sorted([i[0] for i in input_characters_count_select])
target_characters = sorted([i[0] for i in target_characters_count_select])

# 增加未知字符
input_characters.append('<UNK>')
target_characters.append('<UNK>')

# 获取输入字符集和输出字符集的长度
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

# 计算输入数据和输出数据的最长字符数
# max_encoder_seq_length = max([len(txt) for txt in input_texts])
# max_decoder_seq_length = max([len(txt) for txt in target_texts])
max_encoder_seq_length = 150
max_decoder_seq_length = 260

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# 构建字符集字典
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# 将输入和输出数据都转化为one-hot格式
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    if len(target_text) > 260:
        target_text = target_text[:259] + '\n'
    for t, char in enumerate(input_text[:150]):
        try:
            encoder_input_data[i, t, input_token_index[char]] = 1.
        except:
            encoder_input_data[i, t, input_token_index['<UNK>']] = 1.
    for t, char in enumerate(target_text[:260]):
        # decoder_target_data要比decoder_input_data早一个时间步
        try:
            decoder_input_data[i, t, target_token_index[char]] = 1.
        except:
            decoder_input_data[i, t, target_token_index['<UNK>']] = 1.
        if t > 0:
            try:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            except:
                decoder_target_data[i, t - 1, target_token_index['<UNK>']] = 1.

# 训练模型阶段
# encoder层
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# 保留encoder层的隐藏状态
encoder_states = [state_h, state_c]

# decoder层，并将encoder层的隐藏状态作为初始状态
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

#attention_layer = attention_2.Attention_layer()
#attention_outpus = attention_layer(decoder_outputs)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 训练网络
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs)

# 保存模型
model.save('./model/char_model.h5')

# # 载入预训练好的模型
# model = keras.models.load_model('./model/char_model.h5')

# 推理模型阶段，推理阶段在decoder阶段，会把前一个时刻的隐藏状态也作为下一个时刻的输入
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 字符逆字典
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# 定义解码函数，对输入的序列进行解码
def decode_sequence(input_seq):
    # 对输入序列进行编码，并输出其隐藏状态
    states_value = encoder_model.predict(input_seq)

    # 创建空的输出序列
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # 初始化输出序列的第一个字符，即为\t
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # 获取当前输出的字符
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 当输出字符是\n或句子长度已经达到最大的长度限制，则停止
        if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # 更新输入序列
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 更新状态
        states_value = [h, c]

    return decoded_sentence


# 测试效果
for seq_index in range(00, 10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('学员答案：', input_texts[seq_index])
    print('人工评语：', target_texts[seq_index])
    print('机器评语:', decoded_sentence)
