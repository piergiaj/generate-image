# -*- coding: utf-8 -*-
from __future__ import division
import os
import gzip
from fuel.datasets import Dataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream


import numpy as np
from skimage.io import imread
import cPickle
from skimage.transform import resize
randint = np.random.randint

dictionary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digit': 11, 'is': 12, 'on': 13, 'at': 14, 'left': 15, 'right': 16, 'bottom': 17, 'top': 18, 'of': 19, 'image': 20, '.': 21, 'red':22, 'green':23, 'blue':24, 'yellow':25, 'white':26, 'cyan':27, 'magenta':28}

colors = ['red', 'green', 'blue', 'yellow', 'white', 'cyan', 'magenta']
color_channels = [[0], [1], [2], [0,1], [0,1,2], [1,2], [0,2]]


def create_2digit_mnist_image_leftright(digit1, digit2, color1, color2):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((3,60,60))
    digit1 = digit1.reshape(28,28)
    digit2 = digit2.reshape(28,28)

    w = randint(16,18)
    h = randint(0,4)
    image[color1,w:w+28,h:h+28] = digit1

    h = randint(28,32)
    image[color2,w:w+28,h:h+28] = digit2

    image = image.reshape((3,60*60))

    return image

def create_2digit_mnist_image_topbottom(digit1, digit2,color1, color2):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((3,60,60))
    digit1 = digit1.reshape(28,28)
    digit2 = digit2.reshape(28,28)

    h = randint(16,18)
    w = randint(0,2)
    image[color1,w:w+28,h:h+28] = digit1

    w = randint(30,32)
    image[color2,w:w+28,h:h+28] = digit2

    image = image.reshape((3,60*60))

    return image

def create_1digit_mnist_image_topleft(digit1,color):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((3,60,60))
    digit1 = digit1.reshape(28,28)

    w = randint(0,2)
    h = randint(0,4)
    image[color,w:w+28,h:h+28] = digit1

    image = image.reshape((3,60*60))

    return image

def create_1digit_mnist_image_topright(digit1,color):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((3,60,60))
    digit1 = digit1.reshape(28,28)

    w = randint(0,2)
    h = randint(28,32)
    image[color,w:w+28,h:h+28] = digit1

    image = image.reshape((3,60*60))

    return image

def create_1digit_mnist_image_bottomright(digit1,color):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((3,60,60))
    digit1 = digit1.reshape(28,28)

    w = randint(30,32)
    h = randint(28,32)
    image[color,w:w+28,h:h+28] = digit1

    image = image.reshape((3,60*60))

    return image

def create_1digit_mnist_image_bottomleft(digit1,color):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((3,60,60))
    digit1 = digit1.reshape(28,28)

    w = randint(30,32)
    h = randint(0,4)
    image[color,w:w+28,h:h+28] = digit1

    image = image.reshape((3,60*60))

    return image

class CaptionedMNIST(Dataset):

    def __init__(self, banned, bs=64, num=10000, dataset='train', **kwargs):
        self.provides_sources = ('features','captions', 'mask')
        super(CaptionedMNIST, self).__init__(**kwargs)
        self.num_examples = num+(num%bs)
        self.bs = bs

        self.example_iteration_scheme = SequentialExampleScheme(
            self.num_examples)
        self.index = -1
        self.done = False
        self.banned = banned
        f = gzip.open('mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        if dataset == 'train':
            self.labels = train_set[1]
            self.data = train_set[0]
        elif dataset == 'valid':
            self.labels = valid_set[1]
            self.data = valid_set[0]
        elif dataset == 'test':
            self.labels = test_set[1]
            self.data = test_set[0]
        print self.labels.shape

        self.images = np.zeros((bs,3,60*60)).astype('float32')
        self.captions = np.zeros((bs,14)).astype(int)
        self.mask = np.ones((bs,14)).astype(int)


    def reset(self, state):
        self.index = 0
        self.close(state)
        return self.open()    

    def sent2matrix(self, sentence):
        words = sentence.split()
        m = np.int32(np.zeros((1, 14))) 
        
        for i in xrange(len(words)):
            m[0,i] = dictionary[words[i]]
            
        return m

    def get_data(self, state=None, request=None):
        if self.index >= self.num_examples:
            self.done = False
            raise StopIteration
        #bs = request[-1] - request[0]+1
        bs = self.bs
        images = self.images
        images[...] = 0
        captions = self.captions
        captions[...] = 0
        mask = self.mask
        mask[...] = 1
        labels = self.labels
        data = self.data

        for i in range(bs):
            self.index += 1
            while True:
                k = np.random.randint(0,7)
                d1 = np.random.randint(0, self.data.shape[0]-1)
                d2 = np.random.randint(0, self.data.shape[0]-1)
                c1 = np.random.randint(0, len(colors)-1)
                c2 = np.random.randint(0, len(colors)-1)
                # hide some cases
                if k <= 3:
                    if self.labels[d1] == self.banned[k*2] or labels[d2] == self.banned[k*2+1]:
                        continue
                    break
                elif labels[d1] == self.banned[k+4]:
                    continue
                else:
                    break

            if k == 0:
                sentence = 'the %s digit %d is on the left of the %s digit %d .' % (colors[c1], labels[d1], colors[c2], labels[d2])
            elif k == 1:
                sentence = 'the %s digit %d is on the right of the %s digit %d .' % (colors[c2], labels[d2], colors[c1], labels[d1])
            elif k == 2:
                sentence = 'the %s digit %d is at the top of the %s digit %d .' % (colors[c1], labels[d1], colors[c2], labels[d2])
            elif k == 3:
                sentence = 'the %s digit %d is at the bottom of the %s digit %d .' % (colors[c2], labels[d2], colors[c1], labels[d1])
            elif k == 4:
                sentence = 'the %s digit %d is at the top left of the image .' % (colors[c1], labels[d1])
                mask[i,-1] = 0
            elif k == 5: 
                sentence = 'the %s digit %d is at the bottom right of the image .' % (colors[c1], labels[d1])
                mask[i,-1] = 0
            elif k == 6:
                sentence = 'the %s digit %d is at the top right of the image .' % (colors[c1], labels[d1])
                mask[i,-1] = 0
            elif k == 7:
                sentence = 'the %s digit %d is at the bottom left of the image .' % (colors[c1], labels[d1])
                mask[i,-1] = 0
                
            captions[i, :] = self.sent2matrix(sentence)

            if k == 0 or k == 1:
                images[i,:,:] = create_2digit_mnist_image_leftright(data[d1,:], data[d2,:], color_channels[c1], color_channels[c2])
            elif k == 2 or k == 3:
                images[i,:,:] = create_2digit_mnist_image_topbottom(data[d1,:], data[d2,:],color_channels[c1],color_channels[c2])
            elif k == 4:
                images[i,:,:] = create_1digit_mnist_image_topleft(data[d1,:],color_channels[c1])
            elif k == 5:
                images[i,:,:] = create_1digit_mnist_image_bottomright(data[d1,:],color_channels[c1])
            elif k == 6:
                images[i,:,:] = create_1digit_mnist_image_topright(data[d1,:],color_channels[c1])
            elif k == 7:
                images[i,:,:] = create_1digit_mnist_image_bottomleft(data[d1,:],color_channels[c1])
            
        return (images.reshape((images.shape[0], 3*60*60)), captions, mask)
