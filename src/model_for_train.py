import numpy as np
import mindspore
from mindspore import nn, Tensor, context
from mindspore import ops as P
from mindspore.common import dtype as mstype


class NetWithLoss(nn.Cell):
    def __init__(self, model, batch_size=1):
        super(NetWithLoss, self).__init__()
        
        self.batch_size = batch_size
        self.model = model

        self.cast = P.Cast()
        self.transpose = P.Transpose()
        self.trans_matrix = (1, 0)
        self.cross_entropy = nn.BCELoss(reduction="sum")
        self.reduce_sum = P.ReduceSum()
        # self.l2_loss = Regularization(self.model, weight_decay=1e-3)

    def construct(self, content, sen_len, aspect, solution, batch_first=True):
        """
        content: (batch_size, 80) Float32
        sen_len: (batch_size,) Int32
        aspect: (batch_size,) Float32
        solution: (batch_size, 3) Int32
        """

        content = self.cast(content, mstype.int32)
        aspect = self.cast(aspect, mstype.int32)

        pred = self.model(content, sen_len, aspect)
        # pred = self.cast(pred, mstype.float16)
        label = self.cast(solution, mstype.float32)
        loss = self.cross_entropy(pred, label)

        # batch_i = 0
        # loss = 0
        # while batch_i < self.batch_size:
        #     embedding_sen = content[batch_i:batch_i+1, :]
        #     embedding_aspect = aspect[batch_i:batch_i+1]
        #     sen_length = sen_len[batch_i:batch_i+1]
        #     label = solution[batch_i:batch_i+1, :]

        #     pred = self.model(embedding_sen, sen_length, embedding_aspect)
        #     pred = self.cast(pred, mstype.float16)
        #     label = self.cast(label, mstype.float16)

        #     loss += self.cross_entropy(pred, label)
        #     batch_i = batch_i + 1

        return loss

if __name__ == "__main__":

    from model import AttentionLstm
    from load_dataset import load_dataset
    from config import atae_lstm_cfg as cfg
    context.set_context(device_id=4)
    r = np.load(cfg.word_vector)
    word_vector = r['weight']
    weight = Tensor(word_vector, mstype.float16)

    net = AttentionLstm(cfg, weight)
    net_with_loss = NetWithLoss(net, batch_size=10)

    dataset = load_dataset(input_files=cfg.test_dataset, 
                           batch_size=10)
    for data in dataset.create_dict_iterator():
        content = data['content']
        aspect = data['aspect']
        sen_len = data['sen_len']
        solution = data['solution']
        loss = net_with_loss(content, sen_len, aspect, solution)

        print("loss:", loss)
        assert 1==2
