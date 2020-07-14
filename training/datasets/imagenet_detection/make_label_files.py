import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
import os
import sys

from pexpect import searcher_re

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(
    basedir,
    os.path.pardir,
    os.path.pardir,
    os.path.pardir)))

from re3_utils.util import drawing
from re3_utils.util.im_util import get_image_size

DEBUG = False

def main(label_type):
    wildcard = '/*/' if label_type == 'train' else '/'
    dataset_path = 'data/ILSVRC2012_img_train/'
    annotationPath = dataset_path + 'Annotations/'
    imagePath = dataset_path + 'Data/'

    if not os.path.exists(os.path.join('labels', label_type)):
        os.makedirs(os.path.join('labels', label_type))
    imageNameFile = open('labels/' + label_type + '/image_names.txt', 'w')
    search_path = annotationPath + wildcard + '*.xml'
    print("Searching with " + search_path)
    labels = []
    labels = glob.glob(search_path)
    if(len(labels)== 0):
        print("Did not find any files with search path " + search_path)
        exit(1)
    print("Found label paths in the form of ", labels[0])
    labels.sort()
    images = [label.replace('Annotations', 'Data').replace('xml', 'JPEG') for label in labels]

    bboxes = []
    for ii,imageName in enumerate(images):
        if ii % 100 == 0:
            print('iter %d of %d = %.2f%%' % (ii, len(images), ii * 1.0 / len(images) * 100))
        if not DEBUG:
            imageNameFile.write(imageName + '\n')
        imOn = ii
        label = labels[imOn]
        labelTree = ET.parse(label)
        imgSize = get_image_size(images[imOn])
        area_cutoff = imgSize[0] * imgSize[1] * 0.01
        if DEBUG:
            print('\nimage name\n\n%s\n' % images[imOn])
            image = cv2.imread(images[imOn])
            print('image size', image.shape)
            print(label)
            print(labelTree)
            print(labelTree.findall('object'))
        for obj in labelTree.findall('object'):
            bbox = obj.find('bndbox')
            bbox = [int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text),
                    imOn]
            if (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) < area_cutoff:
                continue
            if DEBUG:
                print('name', obj.find('name').text, '\n')
                print(bbox)
                image = image.squeeze()
                if len(image.shape) < 3:
                    image = np.tile(image[:,:,np.newaxis], (1,1,3))
                drawing.drawRect(image, bbox[:-1], 3, [0, 0, 255])
            bboxes.append(bbox)

        if DEBUG:
            if len(image.shape) == 2:
                image = np.tile(image[:,:,np.newaxis], (1,1,3))
            cv2.imshow('image', image)
            cv2.waitKey(0)

    bboxes = np.array(bboxes)
    if not DEBUG:
        np.save('labels/' + label_type + '/labels.npy', bboxes)

if __name__ == '__main__':
    main('train')
    main('val')

