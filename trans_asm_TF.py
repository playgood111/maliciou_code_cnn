import numpy,scipy.misc, os, array
from PIL import Image
from datasets import dataset_utils
import tensorflow as tf
import os
import os.path
import pandas as pd


train_file = 'trainLabels.csv' 
name='train_from_asm_1d'      
output_directory='./tfrecords'
resize_height=299 
resize_width=299 
shape = (resize_height, resize_width, 1)


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
	             #print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], 1, label))
	             #image_raw = image.tostring()
	             png_string = sess.run(encoded_png, feed_dict={image: img})

	             example = dataset_utils.image_to_tfexample(
	               png_string, 'png', resize_height, resize_width, label-1)
	             writer.write(example.SerializeToString())
    writer.close()
    #files=os.listdir(data_set)
    #with open('%s_%s_image_299.csv'%(data_set, data_type),'wb') as f:
        #f.write('Id,%s\n'%','.join(['%s_%i'%(data_type,x)for x in xrange(89401)]))
        #for cc,x in enumerate(files):
            #if data_type != x.split('.')[-1]:
             #   continue
            #file_id = x.split('.')[0]
            #img = read_image(data_set + '/' +x)



            #f.write('%s,%s\n'%(file_id, ','.join(str(v) for v in tmp)))
            #print "finish..." + file_id
def read_image(filename):
    f = open(filename,'rb')
    ln = os.path.getsize(filename) # length of file in bytes
    width = 299
    rem = ln%width
    a = array.array("B") # uint8 array
    a.fromfile(f,ln-rem)
    f.close()
    g = numpy.reshape(a,(len(a)/width,width))
    g = numpy.uint8(g)
    g.resize((89401,))
    im = numpy.reshape(g,(299,299,1))
    
    #img = Image.fromarray(im)
    #save_name = filename.split('.')[0].split('/')[1]
    #img.save("test_asm_image_299/"+save_name+".jpeg")
    
    return im

if __name__ == '__main__':
    #get_feature(data_set = 'train', data_type = 'bytes')
    get_feature(data_set = 'train', data_type = 'asm')
    #get_feature(data_set = 'test', data_type = 'bytes')
    #get_feature(data_set = 'test', data_type = 'asm')
    print 'DONE asm image features!'

