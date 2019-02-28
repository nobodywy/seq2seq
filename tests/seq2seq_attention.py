from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from sklearn.externals import joblib
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils.vis_utils import plot_model
from keras.models import load_model

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

datset = joblib.load('../data/data_seq.pkl')
word_dict = joblib.load('../data/query_dict_seq.pkl')

encoder_input_data = datset[0]
decoder_input_data = datset[1]
decoder_target_data = datset[2]
input_token_index = word_dict[0]
target_token_index = word_dict[1]

num_encoder_tokens = 601
num_decoder_tokens = 601

max_encoder_seq_length = 150
max_decoder_seq_length = 260

'''

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

plot_model(model, to_file='model2.png',show_shapes=True)

print()

'''


if __name__ == '__main__':

    datset = joblib.load('../data/data_seq.pkl')
    word_dict = joblib.load('../data/query_dict_seq.pkl')

    encoder_input_data = datset[0]
    decoder_input_data = datset[1]
    decoder_target_data = datset[2]
    input_token_index = word_dict[0]
    target_token_index = word_dict[1]


    model = AttentionSeq2Seq(output_dim=latent_dim, hidden_dim=latent_dim, output_length=max_decoder_seq_length,
                     input_shape=(max_encoder_seq_length, num_encoder_tokens))

    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')
    #plot_model(model, to_file='model4.png', show_shapes=True)
    model.fit(x=encoder_input_data,y=decoder_target_data,batch_size=batch_size,epochs=epochs)
    model.save('./model/seq2seq_attention_v2.h5')