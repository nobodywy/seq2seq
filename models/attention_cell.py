from keras import layers
from keras.layers import Recurrent, activations, initializers, regularizers, constraints
from keras.layers.merge import multiply

import keras.backend as K

def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x

class AttentionRNNCell(layers.Layer):
    def __init__(self, units,
                 encoder_ts,

                 encoder_latdim,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.units = units
        self.state_size = units


        #self.x_seq = encoder_H
        self.encoder_ts = encoder_ts  # encoder 's timesteps
        self.encoder_latDim = encoder_latdim  # encoder 's latent dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)



        super(AttentionRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):

        _, self.input_dim = input_shape[0]



        """
                    Matrices for creating the context vector
         """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.units, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)





        '''
        z-gate
        '''

        self.C_z = self.add_weight(shape=(self.input_dim + self.encoder_latDim , self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.units, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units,),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
        Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim + self.encoder_latDim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.W_r = self.add_weight(shape=(self.units, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units,),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
        Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim + self.encoder_latDim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units,),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)


        self.built = True

    def call(self, inputs, states, constants):  #inputs 是embeding  states是前一步h
          #


        # prev_output = states[0]
        # h = K.dot(inputs, self.kernel)
        # output = h + K.dot(prev_output, self.recurrent_kernel)

        # compute attention
        # H (timeSteps,lat_dim)
        # H * W_H
          # h维度



        h_tm = states[0]
        self.x_seq = constants[0]

        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                               input_dim=self.units,
                                               timesteps=self.encoder_ts,
                                               output_dim=self.units)


        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(h_tm, self.encoder_ts)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),  # e_ij = a(s_(i-1),h_j)  where h_j = self._uxpb
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.encoder_ts)
        at /= at_sum_repeated  # Eq(6) vector of size (batchsize, timesteps, 1) ， softmax : length timesteps  from stm

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)  # Eq(5)   length : batchsize*latent_dim

        contextInput = K.concatenate([inputs,context],axis=1)
        #拼接context和input


        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(h_tm, self.W_z)
            + K.dot(contextInput, self.C_z)
            + self.b_z)

        rt = activations.sigmoid(
            K.dot(h_tm, self.W_r)
            + K.dot(contextInput, self.C_r)
            + self.b_r)  # f_t 对应lstm遗忘门，此处没有x_t输入

        t_ht  = activations.tanh(
             K.dot((rt * h_tm), self.U_p)
            + K.dot(contextInput, self.C_p)
            + self.b_p
        )

        ht = (1 - zt) * h_tm + zt * t_ht

        return ht, [ht]



# # Let's use this cell in a RNN layer:
#
# cell = MinimalRNNCell(32)
# x = keras.Input((None, 5))
# layer = RNN(cell)
# y = layer(x)