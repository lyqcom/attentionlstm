import numpy as np
import argparse
import sys
import time
import json
import copy

from mindspore import set_seed
from mindspore import nn, Tensor
from mindspore import Model
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.callback import  LossMonitor, SummaryCollector

from model_for_train import NetWithLoss
from model import AttentionLstm
from load_dataset import load_dataset
from config import atae_lstm_cfg as config


if __name__ == '__main__':

    context.set_context(
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False,
                        #save_graphs_path='./train/graphs/',
                        device_id=5)

    dataset = load_dataset(input_files=config.train_dataset, 
                           batch_size=1)
    step_per_epoch = dataset.get_dataset_size()

    r = np.load(config.word_vector)
    word_vector = r['weight']
    weight = Tensor(word_vector, mstype.float16)

    net = AttentionLstm(config, weight, is_train=True)
    model_with_loss = NetWithLoss(net, batch_size=1)

    epoch_size = 25

    lr = Tensor(nn.warmup_lr(learning_rate=0.02, total_step=step_per_epoch*epoch_size,
                            step_per_epoch=step_per_epoch, warmup_epoch=2), mstype.float32)
    # lr = Tensor(nn.polynomial_decay_lr(learning_rate=0.1, end_learning_rate=1e-6, total_step=step_per_epoch*epoch_size,
    #                           step_per_epoch=step_per_epoch, decay_epoch=2, power=0.5), mstype.float32)

    # optimizer = nn.Adagrad(params=net.trainable_params(),
    #                        learning_rate=lr,
    #                        weight_decay=1e-3)
    optimizer = nn.Momentum(params=net.trainable_params(),
                            learning_rate=lr,
                            momentum=0.9,
                            weight_decay=1e-3)
    # optimizer = nn.SGD(params=net.trainable_params(),
    #                     learning_rate=lr,
    #                     momentum=0.9,
    #                     weight_decay=1e-3)
    # optimizer = nn.Adam(params=model_with_loss.trainable_params(),
    #                     learning_rate=0.01,
    #                     weight_decay=1e-3)

    train_net = nn.TrainOneStepCell(model_with_loss, optimizer)

    model = Model(train_net, amp_level="O2")

    time_cb = TimeMonitor(data_size=step_per_epoch)
    loss_cb = LossMonitor()
    summary_collector = SummaryCollector(summary_dir='./train/summary_dir')
    cb = [time_cb, loss_cb, summary_collector]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch,
                                 keep_checkpoint_max=epoch_size)

    ckpoint_cb = ModelCheckpoint(prefix="atae-lstm", directory="./train/", config=config_ck)
    cb.append(ckpoint_cb)

    print("start train")

    model.train(epoch_size, dataset, callbacks=cb)

    print("train success!")
