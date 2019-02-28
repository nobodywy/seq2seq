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
from keras.layers import Concatenate, Dot, Input, LSTM, Embedding, Dense, Reshape, Bidirectional, RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
import keras.backend as K
import tqdm
import gensim
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# 设置超参数
batch_size = 128  # 批次大小
epochs = 1  # 迭代次数
n_a = 32  # The hidden size of Bi-LSTM
n_s = 128  # The hidden size of LSTM in Decoder
num_samples = 10000  # 训练样本大小
input_word_num = 600  # 设置输入词汇的大小
target_word_num = 600  # 设置输出词汇的大小
max_encoder_seq_length = 135  # 输入句子的最大词汇长度
max_decoder_seq_length = 220  # 输出句子的最大词汇长度
emb_dim = 100
data_path = './data/ans_com.txt'  # 数据路径

# 读取数据，并获取输入数据和输出数据的字符集
input_texts = []  # 输入字符串列表
target_texts = []  # 输出字符串列表
input_words_count = dict()
target_words_count = dict()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in tqdm.tqdm(lines[: min(num_samples, len(lines) - 1)]):
    input_text, target_text = line.split('\t')
    # 对输出序列前后添加'\t'和'\n'标记
    target_text = target_text + '\n'
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

# 字符逆字典
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# 将输入和输出数据都转化为整数序列格式
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype='int')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype='float32')

for i, (input_text, target_text) in tqdm.tqdm(enumerate(zip(input_texts, target_texts))):
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
        try:
            decoder_target_data[i, t] = target_token_index[word]
        except:
            decoder_target_data[i, t] = target_token_index['<UNK>']

# 将decoder_target_data转化为one-hot形式
decoder_target_data = to_categorical(decoder_target_data)


# 自定义softmax函数
def softmax(x, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


# 定义全局网络层对象
repeator = RepeatVector(max_encoder_seq_length)
concatenator = Concatenate(axis=-1)
densor_tanh = Dense(32, activation="tanh")
densor_relu = Dense(1, activation="relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)
decoder_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(target_words), activation=softmax)
reshapor = Reshape((1, len(target_words)))
concator = Concatenate(axis=-1)


# 定义注意力层
def one_step_attention(a, s_prev):
    """
    Attention机制的实现，返回加权后的Context Vector

    @param a: BiRNN的隐层状态
    @param s_prev: Decoder端LSTM的上一轮隐层输出

    Returns:
    context: 加权后的Context Vector
    """

    # 将s_prev复制Tx次
    s_prev = repeator(s_prev)
    # 拼接BiRNN隐层状态与s_prev
    concat = concatenator([a, s_prev])
    # 计算energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # 计算weights
    alphas = activator(energies)
    # 加权得到Context Vector
    context = dotor([alphas, a])

    return context


# 加载预训练好的词向量
input_text_w2c = gensim.models.Word2Vec.load('./data/input_text_w2c.model')

word_to_vec_map = {}
for i in input_words:
    try:
        word_to_vec_map[i] = input_text_w2c[i]
    except:
        print(i)
        continue


def pretrained_embedding_layer(word_to_vec_map, source_vocab_to_int, emb_dim):
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
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # build
    embedding_layer.build((None,))

    # set weights
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


# 获取Embedding layer
embedding_layer = pretrained_embedding_layer(word_to_vec_map, input_token_index, emb_dim)


# 构建模型
def model(Tx, Ty, n_a, n_s, source_vocab_size, target_vocab_size):
    """
    构造模型

    @param Tx: 输入序列的长度
    @param Ty: 输出序列的长度
    @param n_a: Encoder端Bi-LSTM隐层结点数
    @param n_s: Decoder端LSTM隐层结点数
    @param source_vocab_size: 输入语料的词典大小
    @param target_vocab_size: 输出语料的词典大小
    """

    # 定义输入层
    X = Input(shape=(Tx,))
    # Embedding层
    embed = embedding_layer(X)
    # Decoder端LSTM的初始状态
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    # Decoder端LSTM的初始输入
    out0 = Input(shape=(target_vocab_size,), name='out0')
    out = reshapor(out0)

    s = s0
    c = c0

    # 模型输出列表，用来存储翻译的结果
    outputs = []

    # 定义Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(embed)

    # Decoder端，迭代Ty轮，每轮生成一个翻译结果
    for t in range(Ty):
        # 获取Context Vector
        context = one_step_attention(a, s)

        # 将Context Vector与上一轮的翻译结果进行concat
        context = concator([context, reshapor(out)])
        s, _, c = decoder_LSTM_cell(context, initial_state=[s, c])

        # 将LSTM的输出结果与全连接层链接
        out = output_layer(s)

        # 存储输出结果
        outputs.append(out)

    model = Model([X, s0, c0, out0], outputs)

    return model


model = model(max_encoder_seq_length, max_decoder_seq_length, n_a, n_s, len(input_words), len(target_words))
# model.summary()
out = model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001), loss='categorical_crossentropy')

# 初始化各类向量
m = encoder_input_data.shape[0]
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
out0 = np.zeros((m, len(target_words)))
outputs = list(decoder_target_data.swapaxes(0, 1))

# 训练模型
model.fit([encoder_input_data, s0, c0, out0], outputs, epochs=epochs, batch_size=batch_size)

# 保存模型
model.save('./model/seq2seq_with_attn.h5')

# 加载模型
model = load_model('./model/seq2seq_with_attn.h5')

# 预测
pred = model.predict([encoder_input_data[0:1], s0, c0, out0])
pred = np.argmax(pred, axis=-1)

