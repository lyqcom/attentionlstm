import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter


class LSTM(nn.Cell):
    def __init__(self, input_size, hidden_size, has_bias=True, batch_first=False, dropout=0):
        super(LSTM, self).__init__()

        if not 0 <= dropout <= 1:
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_op = nn.Dropout(float(1 - dropout))
        self.has_bias = has_bias

        stdv = 1 / self.hidden_size
        weight = Tensor(np.random.uniform(-stdv, stdv, (hidden_size, input_size + hidden_size)).astype(np.float16))
        bias = Tensor(np.random.uniform(-stdv, stdv, (hidden_size, 1)).astype(np.float16))  # batch_size = 1
        self.Wi = Parameter(weight, name='Wi')
        self.Wo = Parameter(weight, name='Wo')
        self.Wf = Parameter(weight, name='Wf')
        self.Wc = Parameter(weight, name='Wc')
        self.bi = Parameter(bias, name='bi')
        self.bo = Parameter(bias, name='bo')
        self.bf = Parameter(bias, name='bf')
        self.bc = Parameter(bias, name='bc')

        self.concat_0 = P.Concat(axis=0)
        self.concat_1 = P.Concat(axis=1)
        self.squeeze_0 = P.Squeeze(axis=0)
        self.transpose = P.Transpose()
        self.tensor_order = (1, 0, 2)
        self.matrix_order = (1, 0)
        self.reshape = P.Reshape()
        self.sigmoid = P.Sigmoid()
        self.tanh = P.Tanh()
        self.matmul = P.MatMul(False, True)  # !

    def lstm_cell(self, input_x, hidden):
        """
        input: [batch_size, input_size]
        hidden: tuple([batch_size, hidden_size], [batch_size, hidden_size])
        w_ih : [4 * hidden_size, hidden_size + input_size]
        b_ih : [4 * hidden_size]
        """
        hx, cx = hidden
        batch_size = hx.shape[0]
        cx = self.reshape(cx, (self.hidden_size, batch_size))

        inputs = self.concat_1((hx, input_x))
        f_t = self.sigmoid(self.matmul(self.Wf, inputs) + self.bf)
        i_t = self.sigmoid(self.matmul(self.Wi, inputs) + self.bi)
        o_t = self.sigmoid(self.matmul(self.Wo, inputs) + self.bo)
        cy = f_t * cx + i_t * self.tanh(self.matmul(self.Wc, inputs) + self.bc)
        hy = o_t * self.tanh(cy)

        hy = self.reshape(hy, (batch_size, self.hidden_size))
        cy = self.reshape(cy, (batch_size, self.hidden_size))

        return hy, cy

    def construct(self, input_x, hidden, seq_length):
        """
        inputx : [seq_length, batch_size, input_size]
        hidden : ([1, batch_size, hidden_size], [1, batch_size, hidden_size])
        seq_length : [batch_size,]
        """
        if self.batch_first:
            input_x = self.transpose(input_x, self.tensor_order)

        h, c = hidden[0][0], hidden[1][0]
        time_step = input_x.shape[0]

        seq_length = P.Cast()(seq_length, mindspore.int32)
        seq_length = P.BroadcastTo((self.hidden_size, -1))(seq_length)
        seq_length = P.Transpose()(seq_length, (1, 0))
        
        zero_output = P.ZerosLike()(h)
        outputs = []
        state_t = (h, c)
        t = 0
        while t < time_step:
            h_t = self.lstm_cell(self.squeeze_0(input_x[t:t+1]), state_t)
            seq_cond = seq_length > t
            state_t_0 = P.Select()(seq_cond, h_t[0], state_t[0])
            state_t_1 = P.Select()(seq_cond, h_t[1], state_t[1])
            output = P.Select()(seq_cond, h_t[0], zero_output)
            state_t = (state_t_0, state_t_1)
            outputs.append(output)
            t += 1
        outputs = P.Stack()(outputs)
        if self.batch_first:
            outputs = self.transpose(outputs, self.tensor_order)
        return outputs, state_t

# if __name__ == "__main__":
#     from mindspore import context
#     context.set_context(device_id=5)
#     lstm = LSTM(8, 8, batch_first=True, has_bias=True)

#     inputs = Tensor(np.random.rand(1, 6, 8).astype(np.float16))
#     h0 = Tensor(np.zeros((1, 1, 8)).astype(np.float16))
#     c0 = Tensor(np.zeros((1, 1, 8)).astype(np.float16))
#     seq_len = Tensor(np.random.randint(0, 5, (1)))

#     import time
#     time_start = time.time()
#     out, (hn, cn) = lstm(inputs, (h0, c0), seq_len)
#     print("out:", out.shape)
#     print("out:\n", out)
#     print("h_n:\n", hn)
#     time_end = time.time()
#     print('my lstm time cost',time_end-time_start,'s')

