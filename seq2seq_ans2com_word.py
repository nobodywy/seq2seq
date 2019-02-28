'''
seq2seq基于词汇级
'''
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import pickle
import jieba
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
import os
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

# 设置超参数
New_model = True #是否初次训练
model_path = './model/word_model_triple_lstm.h5'  #model path
new_model_path =  './model/word_triple_lstm'

batch_size = 64  # 批次大小
epochs = 50  # 迭代次数
latent_dim = 256  # LSTM隐藏单元的数量
num_samples = 15000  # 训练样本大小
input_word_num = 2000  # 设置输入词汇的大小
target_word_num = 4000  # 设置输出词汇的大小
max_encoder_seq_length = 135  # 输入句子的最大词汇长度
max_decoder_seq_length = 220  # 输出句子的最大词汇长度
embedding_dim = 100  # embedding层维数
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
decoder_target_data = to_categorical(decoder_target_data)


#创建embedding layer
import gensim
# 加载预训练好的词向量
input_text_w2c = gensim.models.Word2Vec.load('./data/input_text_w2c.model')
target_text_w2c = gensim.models.Word2Vec.load('./data/target_text_w2c.model')

word_to_vec_map = {}
for i in input_words:
    try:
        word_to_vec_map[i] = input_text_w2c[i]
    except:
        print(i)
        continue

target_to_vec_map = {}
for i in target_words:
    try:
        target_to_vec_map[i] = target_text_w2c[i]
    except:
        print(i)
        continue


def pretrained_embedding_layer(word_to_vec_map, source_vocab_to_int, emb_dim,trainable = False):
    """
    构造Embedding层并加载预训练好的词向量

    @param word_to_vec_map: 单词到向量的映射
    @param word_to_index: 单词到数字编码的映射
    @param emb_dim：embedding维度
    """

    vocab_len = len(source_vocab_to_int)

    # 初始化embedding矩阵
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # 用词向量填充embedding矩阵
    for word, index in source_vocab_to_int.items():
        word_vector = word_to_vec_map.get(word, np.zeros(emb_dim))
        emb_matrix[index, :] = word_vector

    # 定义Embedding层，并指定不需要训练该层的权重
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=trainable)

    # build
    embedding_layer.build((None,))

    # set weights
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


# 获取Embedding layer
emb_dim = embedding_dim
embedding_layer = pretrained_embedding_layer(word_to_vec_map, input_token_index, emb_dim)
target_emedding_layer = pretrained_embedding_layer(target_to_vec_map, target_token_index, emb_dim,trainable=True)

# 训练模型阶段
def build_basic_model():
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    #encoder_embedding = Embedding(num_encoder_tokens, embedding_dim, name='encoder_embedding')(encoder_inputs)
    encoder_embedding = embedding_layer(encoder_inputs)
    encoder_lstm_1 = LSTM(latent_dim, dropout=0.1, return_state=True, return_sequences=True, name='encoder_lstm_1')
    encoder_lstm_1_out, *encoder_states_1 = encoder_lstm_1(encoder_embedding)
    encoder_lstm_2 = LSTM(latent_dim, dropout=0.1, return_state=True, return_sequences=True, name='encoder_lstm_2')
    encoder_lstm_2_out, *encoder_states_2 = encoder_lstm_2(encoder_lstm_1_out)
    encoder_lstm_3 = LSTM(latent_dim, dropout=0.2, return_state=True, return_sequences=True, name='encoder_lstm_3')
    _, *encoder_states_3 = encoder_lstm_3(encoder_lstm_2_out)

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    #decoder_embedding = Embedding(num_decoder_tokens, embedding_dim, name='decoder_embedding')(decoder_inputs)
    decoder_embedding = target_emedding_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, dropout=0.2, return_state=True, return_sequences=True, name='decoder_lstm')
    rnn_outputs, *decoder_states = decoder_lstm(decoder_embedding, initial_state=encoder_states_3)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(rnn_outputs)

    basic_model = Model([encoder_inputs, decoder_inputs], [decoder_outputs])
    basic_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return basic_model

# # 回调函数
# callback_list = [ModelCheckpoint('./model/basic_model_best.h5', save_best_only=True)]

# 获取模型
if(New_model):
    basic_model = build_basic_model()
else:
    basic_model = load_model(model_path)
# 训练
epochs = 10
for i in range(0,5):
    basic_model_hist = basic_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                       batch_size=batch_size, epochs=epochs)
    # 保存模型
    basic_model.save(new_model_path+'_epo_'+str(int(i)*10+epochs)+'.h5')




#测试
#basic_model = load_model('model/word_triple_lstm_epo_10.h5')

'''

# 绘制模型结构
plot_model(basic_model, to_file='./model/word_model.png', show_shapes=True, show_layer_names=True)


# 建立推理模型
def build_basic_inference_model(model_path):
    model = load_model(model_path)

    # encoder
    encoder_inputs = Input(shape=(None,))
    # encoder_embedding
    encoder_embedding = model.get_layer('encoder_embedding')(encoder_inputs)
    # get encoder states
    _, *encoder_states = model.get_layer('encoder_lstm')(encoder_embedding)
    encoder_model = Model(encoder_inputs, encoder_states)

    # decoder
    # decoder inputs
    decoder_inputs = Input(shape=(None,))
    # decoder input states
    decoder_state_h = Input(shape=(latent_dim,))
    decoder_state_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_h, decoder_state_c]

    # decoder embedding
    decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)
    # get rnn outputs and decoder states
    rnn_outputs, *decoder_states = model.get_layer('decoder_lstm')(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_outputs = model.get_layer('decoder_dense')(rnn_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


encoder_model, decoder_model = build_basic_inference_model('./model/word_model.h5')

# 字符逆字典
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


# 定义解码函数，对输入的序列进行解码
def decode_sequence(input_seq,encoder_model,decoder_model):
    # 对输入序列进行编码，并输出其隐藏状态
    states_value = encoder_model.predict(input_seq)

    # 创建空的输出序列
    target_seq = np.zeros((1, 1), dtype='int')
    # 初始化输出序列的第一个字符，即为\t
    target_seq[0][0] = target_token_index['\t']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 获取当前输出的字符
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # 当输出字符是\n或句子长度已经达到最大的长度限制，则停止
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

            # 更新输入序列
            target_seq = np.zeros((1, 1))
            target_seq[0][0] = sampled_token_index

        # 更新状态
        states_value = [h, c]

    return ''.join(decoded_sentence)


# 测试效果
for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq,encoder_model,decoder_model)
    print('-')
    print('学员答案：', input_texts[seq_index])
    print('人工评语：', target_texts[seq_index])
    print('机器评语:', decoded_sentence)
'''