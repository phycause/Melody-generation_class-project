# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_MIDItxt(filename):
  with open(filename, "r") as text_file:
    return list(map(int, text_file.read().split(' ')))

def ptb_raw_data(data_path=None):

  ##定義0~88為pitch，89為延續上一個音，90為沒有音
  midi_idx_path = os.path.join(data_path, "melodygen.midi_idx.txt")
  train_path = os.path.join(data_path, "melodygen.train.txt")
  valid_path = os.path.join(data_path, "melodygen.valid.txt")
  test_path = os.path.join(data_path, "melodygen.test.txt")

  midi_idx = _read_MIDItxt(midi_idx_path) #使用train.txt來做word to id
  train_data = _read_MIDItxt(train_path) #如果本來midi就已經編號好了 這部分不用做
  valid_data = _read_MIDItxt(valid_path)
  test_data = _read_MIDItxt(test_path)
  midi_idx_numbers = len(midi_idx)
  return train_data, valid_data, test_data, midi_idx_numbers


def ptb_producer(raw_data, batch_size, num_steps, name=None):

  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
