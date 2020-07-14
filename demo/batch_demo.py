import cv2
import glob
import numpy as np
import sys
import os.path

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
import tensorflow as tf
from nets.mobilenet import mobilenet_v2
#from slim.nets.mobilenet import mobilenet_v2

#from tensorflow.python.saved_model import tag_constants
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

    images = tf.expand_dims(image, 0)
    images = tf.cast(images, tf.float32) / 128. - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (192, 192))

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        logits, endpoints = mobilenet_v2.mobilenet(images)
    ema = tf.train.ExponentialMovingAverage(0.999)
    vars = ema.variables_to_restore()
    print("Vars: ", vars)
    saver = tf.train.Saver(vars)
    saver.restore(sess, "/home/waechter/repos/tf-models/research/slim/nets/mobilenet/checkpoint/mobilenet_v2_1.0_192.ckpt")
    print("Model restored.")


from tracker import re3_tracker

if not os.path.exists(os.path.join(basedir, 'data')):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, 'data.tar.gz'))
    tar.extractall(path=basedir)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 640, 480)
tracker = re3_tracker.Re3Tracker()
image_paths = sorted(glob.glob(os.path.join(
    os.path.dirname(__file__), 'data', '*.jpg')))
initial_bbox = [190, 158, 249, 215]
# Provide a unique id, an image/path, and a bounding box.
tracker.track('ball', image_paths[0], initial_bbox)
print('ball track started')
for ii,image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    # Tracker expects RGB, but opencv loads BGR.
    imageRGB = image[:,:,::-1]
    if ii < 100:
        # The track alread exists, so all that is needed is the unique id and the image.
        bbox = tracker.track('ball', imageRGB)
        color = cv2.cvtColor(np.uint8([[[0, 128, 200]]]),
            cv2.COLOR_HSV2RGB).squeeze().tolist()
        cv2.rectangle(image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color, 2)
    elif ii == 100:
        # Start a new track, but continue the first as well. Only the new track needs an initial bounding box.
        bboxes = tracker.multi_track(['ball', 'logo'], imageRGB, {'logo' : [399, 20, 428, 45]})
        print('logo track started')
    else:
        # Both tracks are started, neither needs bounding boxes.
        bboxes = tracker.multi_track(['ball', 'logo'], imageRGB)
    if ii >= 100:
        for bb,bbox in enumerate(bboxes):
            color = cv2.cvtColor(np.uint8([[[bb * 255 / len(bboxes), 128, 200]]]),
                cv2.COLOR_HSV2RGB).squeeze().tolist()
            cv2.rectangle(image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color, 2)
    cv2.imshow('Image', image)
    cv2.waitKey(1)
