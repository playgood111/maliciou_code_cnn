import numpy,scipy.misc, os, array
from PIL import Image
from datasets import dataset_utils
import tensorflow as tf
import os
import os.path
import pandas as pd
import cv2

train_file = 'trainLabels.csv' 
name='train_bytes_image_299_3D'      
output_directory='./tfrecords'
resize_height=299 
resize_width=299 
shape = (resize_height, resize_width, 3)


def load_file(examples_list_file):
    #lines = np.genfromtxt(examples_list_file, delimiter=" ", dtype=[('col1', 'S120'), ('col2', 'i8')])
    trainLabels = pd.read_csv(examples_list_file)

    examples = trainLabels.Id
    labels = trainLabels.Class
    #for example, label in lines:
    #    examples.append(example)
    #    labels.append(label)
    return numpy.asarray(examples), numpy.asarray(labels), len(labels)
def get_feature(data_set = 'train', data_type = 'bytes'):
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
    _examples, _labels, examples_num = load_file(train_file)
    filename = output_directory + "/" + name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)

    with tf.Graph().as_default():

        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)
        with tf.Session('') as sess:
            for i, [example, label] in enumerate(zip(_examples, _labels)):
                print('No.%d' % (i))
                img = read_image(data_set + '/' +example+'.'+data_type)
                png_string = sess.run(encoded_png, feed_dict={image: img})

                example = dataset_utils.image_to_tfexample(
                    png_string, 'png', resize_height, resize_width, label-1)
                writer.write(example.SerializeToString())
    writer.close()
 
def read_image(filename):
    image = cv2.imread(filename,cv2.CV_LOAD_IMAGE_COLOR)
    image = cv2.resize(image, (resize_height, resize_width))


    return image

if __name__ == '__main__':
    #get_feature(data_set = 'train', data_type = 'bytes')
    get_feature(data_set = 'train_bytes_image_299', data_type = 'jpeg')
    #get_feature(data_set = 'test', data_type = 'bytes')
    #get_feature(data_set = 'test', data_type = 'asm')
    print 'DONE asm image features!'

