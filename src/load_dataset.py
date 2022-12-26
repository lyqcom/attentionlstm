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
"""Dataset loader to feed into model."""
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as deC

from mindspore import context, Tensor
import mindspore
from mindspore import ops as P


def load_dataset(input_files, batch_size, sink_mode=False,
                  rank_size=1, rank_id=0, shuffle=True, drop_remainder=True):
    """
    Load dataset according to passed in params.

    Args:
        input_files (list): Data files.
        batch_size (int): Batch size.
        sink_mode (bool): Whether enable sink mode.
        rank_size (int): Rank size.
        rank_id (int): Rank id.
        shuffle (bool): Whether shuffle dataset.
        drop_remainder (bool): Whether drop the last possibly incomplete batch.
        is_translate (bool): Whether translate the text.

    Returns:
        Dataset, dataset instance.
    """
    if not input_files:
        raise FileNotFoundError("Require at least one dataset.")

    if not isinstance(sink_mode, bool):
        raise ValueError("`sink` must be type of bool.")

    for datafile in input_files:
        print(f" | Loading {datafile}.")

    data_set = ds.MindDataset(
        input_files, columns_list=[
        "content", "sen_len", "aspect", "solution"],
        shuffle=False, num_shards=rank_size, shard_id=rank_id,
        num_parallel_workers=8
        )

    if shuffle:
        data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())

    fp_cast_op = deC.TypeCast(mstype.float32)
    int_cast_op = deC.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="content", operations=fp_cast_op, num_parallel_workers=8)
    data_set = data_set.map(input_columns="sen_len", operations=int_cast_op, num_parallel_workers=8)
    data_set = data_set.map(input_columns="aspect", operations=fp_cast_op, num_parallel_workers=8)
    data_set = data_set.map(input_columns="solution", operations=int_cast_op, num_parallel_workers=8)


    ori_dataset_size = data_set.get_dataset_size()
    print(f" | Dataset size: {ori_dataset_size}.")

    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set


if __name__ == "__main__":
    context.set_context(device_target="Ascend", device_id=2)

    train_dataset = load_dataset(input_files="../dataset/test.mindrecord", 
                           batch_size=1, 
                           sink_mode=False,
                           rank_size=1, 
                           rank_id=0, 
                           shuffle=True, 
                           drop_remainder=True)

    argmax = P.Argmax()
    for data in train_dataset.create_dict_iterator():
        content = data['content']
        sen_len = data['sen_len']
        aspect = data['aspect']
        solution = data['solution']
        print("content:", content.shape)
        print("sen_len:", sen_len.shape)
        print("aspect:", aspect.shape)
        print("solution:", solution.shape)
        print(content.dtype)
        print(sen_len.dtype)
        print(aspect.dtype)
        print(solution.dtype)

        # p = argmax(solution)
        # print(p)
        assert 1==2
        print("\n-------------\n")
        

    # dataset_helper = mindspore.DatasetHelper(train_dataset, dataset_sink_mode=True)
    # dataset = dataset_helper.iter.dataset
    # dataset_types, dataset_shapes = dataset_helper.types_shapes()
    # queue_name = dataset.__transfer_dataset__.queue_name

    # get_next = P.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)


    # content, sen_len, aspect, solution = get_next()
    # reshape = P.Reshape()
    # #solution = reshape(solution, (-1, 3))
    # solution = Tensor(solution.asnumpy().reshape(-1,3).astype(np.int32))
    # p = np.argmax(solution)
    # print(p)



    
