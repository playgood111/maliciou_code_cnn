# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts MNIST data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf
import pandas as pd
from datasets import dataset_utils


image_file = 'train_asm_image_299.csv'
train_file = 'trainLabels.csv' 
name='train_asm_1d'      
output_directory='./tfrecords'
_IMAGE_SIZE=299 
#resize_width=299 
_IMAGE_NUM=10868
_IMAGE_CHANEL=1
shape = (_IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANEL)



def _add_to_tfrecord(tfrecord_writer):
  """Loads data from the binary MNIST files and writes files to a TFRecord.

  Args:
    data_filename: The filename of the MNIST images.
    labels_filename: The filename of the MNIST labels.
    num_images: The number of images in the dataset.
    tfrecord_writer: The TFRecord writer to use for writing.
  """


  data_ori = pd.read_csv(image_file)
  labels_ori = pd.read_csv(train_file)
  data = pd.merge(data_ori,labels_ori,on='Id')
  labels = data.Class
  data.drop(["Class","Id"], axis=1, inplace=True)
  train_data = data.values[:,:]
  images = train_data.reshape(_IMAGE_NUM,_IMAGE_SIZE,_IMAGE_SIZE,_IMAGE_CHANEL)


  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    encoded_png = tf.image.encode_png(image)

    with tf.Session('') as sess:
      for j in range(_IMAGE_NUM):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, _IMAGE_NUM))
        sys.stdout.flush()

        png_string = sess.run(encoded_png, feed_dict={image: images[j]})

        example = dataset_utils.image_to_tfexample(
            png_string, 'png', _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
        tfrecord_writer.write(example.SerializeToString())





  # First, process the training data:
if not os.path.exists(output_directory) or os.path.isfile(output_directory):
  os.makedirs(output_directory)
filename = output_directory + "/" + name + '.tfrecords'
with tf.python_io.TFRecordWriter(filename) as tfrecord_writer:
  #data_filename = os.path.join(dataset_dir, _TRAIN_DATA_FILENAME)
  #labels_filename = os.path.join(dataset_dir, _TRAIN_LABELS_FILENAME)
  _add_to_tfrecord(tfrecord_writer)

  