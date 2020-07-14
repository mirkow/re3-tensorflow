
import tensorflow as tf
from nets.mobilenet import mobilenet_v2
from datasets import imagenet
import numpy as np
from datetime import datetime
import time

import tensorflow as tf
#import tf.keras as keras

with tf.Session() as sess:
    #saver = tf.train.import_meta_graph('/home/waechter/repos/tf-models/research/slim/nets/mobilenet/checkpoint/mobilenet_v2_1.0_192.ckpt.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('/home/waechter/repos/tf-models/research/slim/nets/mobilenet/checkpoint/'))
    #all_vars = tf.get_collection('vars')
    #for v in all_vars:
#        v_ = sess.run(v)
#        print(v_)
#    saver = tf.train.Saver()
#    saver.restore(sess,
#              "/home/waechter/repos/tf-models/research/slim/nets/mobilenet/checkpoint/mobilenet_v2_1.0_192.ckpt")
    #tf.saved_model.loader.load(sess, [tag_constants.SERVING], "/home/waechter/repos/tf-models/research/slim/nets/mobilenet/checkpoint/")

    file_input = tf.placeholder(tf.string, ())

    image = tf.image.decode_jpeg(tf.read_file(file_input))
    image_size = 128
    depth_multiplier = 0.5
    images = tf.expand_dims(image, 0)
    images = tf.cast(images, tf.float32) / 128. - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (image_size, image_size))

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
        #logits, endpoints = mobilenet_v2.mobilenet(images, depth_multiplier=depth_multiplier)
        logits, endpoints = mobilenet_v2.mobilenet_v2_050(images)
    ema = tf.train.ExponentialMovingAverage(0.999)
    vars = ema.variables_to_restore()
    #print("Vars: ", vars)
    saver = tf.train.Saver(vars)
    #saver.restore(sess, "/home/waechter/repos/tf-models/research/slim/nets/mobilenet/checkpoint/mobilenet_v2_1.0_192.ckpt")
    saver.restore(sess, "/home/waechter/repos/tf-models/research/slim/nets/mobilenet/checkpoint/mobilenet_v2_" +str(depth_multiplier) + "_" + str(image_size) + ".ckpt")
    print("Model restored.")

    x = endpoints['Predictions'].eval(feed_dict={file_input: 'panda.jpg'})
    print("Endpoints: ", endpoints.keys())
    label_map = imagenet.create_readable_names_for_imagenet_labels()
    #print("Labels: ", label_map)
    sorted_indices = np.argsort(x)
    print("Predictions: ", sorted_indices, sorted_indices[0,-1])
    print("Top 1 prediction: ", x.argmax(), label_map[x.argmax()], x.max())
    print("Top 5 predictions: ", [ (label_map[sorted_indices[0,-i-1]], x[0,sorted_indices[0,-i-1]]) for i in range(0,5) ])
    x = endpoints['layer_18'].eval(feed_dict={file_input: 'panda.jpg'})
    print("X18: ", x.shape)
    start = time.time()
    iterations = 100
    for i in range(0,iterations):
        x = endpoints['layer_17'].eval(feed_dict={file_input: 'panda.jpg'})
    end = time.time()
    print("Duration per iteration: ", (end-start)/iterations)
    print("X17: ", x.shape)
# Define the Keras TensorBoard callback.
# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
#print(tf.summary.tensor_summary("mobilenet_v2", endpoints['Predictions']))
    file_writer = tf.summary.FileWriter('logs', sess.graph)
    converter = tf.lite.TFLiteConverter.from_session(sess, [images], [endpoints['Predictions']])
    tflite_model = converter.convert()
    #open("converted_model.tflite", "wb").write(tflite_model)