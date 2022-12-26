
from mindspore import ops as P
from mindspore import nn
from mindspore.common import dtype as mstype


class Infer(nn.Cell):
    def __init__(self, model, batch_size=1):
        super(Infer, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.cast = P.Cast()
        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)

    def construct(self, content, sen_len, aspect, batch_first=True):
        content = self.cast(content, mstype.int32)
        aspect = self.cast(aspect, mstype.int32)
        pred = self.model(content, sen_len, aspect)
        pred = self.cast(pred, mstype.float32)
        return pred
