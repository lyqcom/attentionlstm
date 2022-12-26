import numpy as np

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Model
from mindspore import context, nn, Tensor 
from mindspore import ops as P
from mindspore.common import dtype as mstype

from config import atae_lstm_cfg as config
from model import AttentionLstm
from model_for_test import Infer
from load_dataset import load_dataset


if __name__ == '__main__':
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        device_id=7)

    config.test_dataset = config.test_dataset

    dataset = load_dataset(input_files=config.test_dataset, 
                           batch_size=1)

    r = np.load(config.word_vector)
    word_vector = r['weight']
    weight = Tensor(word_vector, mstype.float16)

    net = AttentionLstm(config, weight, is_train=False) 

    max_acc = 0
    for i in range(1,26):

        config.existed_ckpt = '/root/xidian_wks/lhh/new_ATAE/src/train/atae-lstm_3-'+str(
                                i)+'_299.ckpt'

        model_path = config.existed_ckpt
        ms_ckpt = load_checkpoint(model_path)
        load_param_into_net(net, ms_ckpt)
        infer = Infer(net, batch_size=1)

        model = Model(infer)

        correct = 0
        count = 0

        for batch in dataset.create_dict_iterator():
            content = batch['content']
            sen_len = batch['sen_len']
            aspect = batch['aspect']
            solution = batch['solution']

            pred = model.predict(content, sen_len, aspect)

            polarity_pred = np.argmax(pred.asnumpy())
            polarity_label = np.argmax(solution.asnumpy())
            # print(polarity_pred, polarity_label)

            if polarity_pred == polarity_label:
                correct += 1
            count += 1
        acc = correct / count
        print("ckpt_", i, "---accuracy:", acc, "---")
        if acc > max_acc:
            max_acc = acc
    print("max_acc:", max_acc)
