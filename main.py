import os, sys, random
import tensorflow as tf
import numpy as np
from skimage.io import imshow, imread
from skimage.color import hsv2rgb
import matplotlib.pyplot as plt

from models.VGG16FlowSearch import VGG16FlowSearch

root = os.path.dirname(os.path.realpath(__file__))

# Config
tf.flags.DEFINE_boolean('debug', False, 'Debug mode')
tf.flags.DEFINE_string('image1', 'data/training/image_2/000055_10.png', 'Image 1')
tf.flags.DEFINE_string('image2', 'data/training/image_2/000055_11.png', 'Image 2')
tf.flags.DEFINE_integer('ymin', -64, 'Disparity range minimum in y-Direction')
tf.flags.DEFINE_integer('ymax', 64, 'Disparity range maximum in y-Direction')
tf.flags.DEFINE_integer('xmin', -64, 'Disparity range minimum in x-Direction')
tf.flags.DEFINE_integer('xmax', 64, 'Disparity range maximum in x-Direction')
tf.flags.DEFINE_integer('ystep', 16, 'Disparity block size in y-Direction')
tf.flags.DEFINE_integer('xstep', 16, 'Disparity block size in x-Direction')


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    rho = rho / np.max(rho)
    return (rho, phi)

def vis_flow(image):
    rho, phi = cart2pol(image[:, :, 0], image[:, :, 1])
    return hsv2rgb(np.stack((phi, rho, np.ones_like(rho)), axis=-1))

def main(_):
    FLAGS = tf.flags.FLAGS
    if FLAGS.debug:
        for k, v in FLAGS.__flags.items():
            print("{}: {}".format(k.upper(), v.value))
        print("")

    model = VGG16FlowSearch()
    im1 = imread(os.path.join(root, FLAGS.image1))
    im2 = imread(os.path.join(root, FLAGS.image2))
    flow = model.infer(im1, im2, d_range=[[FLAGS.ymin,FLAGS.ymax],[FLAGS.xmin,FLAGS.xmax]],
                       step=[FLAGS.ystep,FLAGS.xstep])

    plt.figure(figsize=(26, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.subplot(1, 2, 2)
    plt.imshow(im2)

    plt.figure(figsize=(16, 6))
    plt.imshow(vis_flow(flow))

    plt.figure(figsize=(16, 6))
    plt.imshow(flow[:, :, 0], cmap='plasma')
    plt.colorbar()
    plt.figure(figsize=(16, 6))
    plt.imshow(flow[:, :, 1], cmap='plasma')
    plt.colorbar()

if __name__ == '__main__':
    tf.app.run()