# AttentionLSTM

## 目录



## AttentionLSTM概述

给定一个句子和句子中出现的某个aspect，aspect-level 情感分析的目标是分析出这个句子在给定aspect上的情感倾向。

例如：great food but the service was dreadful! 在aspect “food”上，情感倾向为正，在aspect “service”上情感倾向为负。Aspect level的情感分析相对于document level来说粒度更细。

论文： Attention-based LSTM for Aspect-level Sentiment Classification  [https://www.aclweb.org/anthology/D16-1058.pdf]

## 模型架构

AttentionLSTM模型的输入由词向量和Aspect向量组成，LSTM是单向单层的结构，输出的状态向量再和Aspect向量concatenate，并输入到Attention结构中计算Attention权值。最后拿Attention权值和LSTM输出的状态向量一起计算情感极性。

## 数据集

- semEval 2014 Task 4
- 预训练词向量：Glove



## 环境要求

- 硬件：Ascend处理器
- 框架：mindspore

## 快速入门

在Ascend处理器上运行

```python
cd src
python train_atae_lstm.py
```



## 脚本说明

```shell
.
├── AttentionLSTM
    ├── README.md               # AttnetionLSTM相关说明
    ├── script
    │   ├── run_eval.sh  	    # 
    │   └── run_train.sh    	# 
    ├── src
    │   ├── config.py           # 参数配置
    │   ├── eval_atae_lstm.py   # 推理脚本
    │   ├── load_dataset.py     # 加载数据集
    │   ├── model.py            # 模型结构
    │   ├── model_for_test.py   # 模型推理 
    │   ├── model_for_train.py  # 模型训练
    │   ├── my_utils.py         # LSTM配置 
    │   ├── rnn_cells.py        # LSTMCell
    │   ├── train_atae_lstm.py  # 训练脚本
    │   └── rnns.py             # LSTM
    ├── eval.py                 # GPU、CPU和Ascend的评估脚本
    └── train.py                # GPU、CPU和Ascend的训练脚本
```





