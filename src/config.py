#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
network config
"""
from easydict import EasyDict as edict

atae_lstm_cfg = edict({
    'batch_size': 1,
    'dim_hidden': 300,
    'rseed': 123456,
    'dim_word': 300,
    'dim_aspect': 100,
    'optimizer': 'ADAGRAD',
    'regular': 0.001,
    'vocab_size': 5177,
    'dropout_prob': 0.5,
    'aspect_num': 5,
    'train_dataset': '/root/xidian_wks/lhh/new_ATAE/mine_dataset/train.mindrecord',
    'test_dataset': '/root/xidian_wks/lhh/new_ATAE/mine_dataset/test.mindrecord',
    'word_vector': '/root/xidian_wks/lhh/glove_file/weight_new.npz',
    'grained': 3,
    'lr': 0.002,
    'lr_word_vector': 0.1,
    'epoch': 25
})