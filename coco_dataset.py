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

dictionary = cPickle.load(open('coco/dictionary.pkl','rb'))


class MSCoco(Dataset):

    def __init__(self, bs=64, num=82611, dataset='train', img_size=32, lang_N=57, **kwargs):
        self.provides_sources = ('features','captions', 'mask')
        super(MSCoco, self).__init__(**kwargs)
        self.num_examples = num+(num%bs)
        self.bs = bs
        self.num = num
        self.lang_N = lang_N

        self.example_iteration_scheme = SequentialExampleScheme(
            self.num_examples)
        self.index = 0
        self.done = False

        self.imgs = np.load('coco/'+dataset+'-images-'+str(img_size)+'x'+str(img_size)+'.npy')
        self.imgs = self.imgs.reshape((num, 3,img_size*img_size))
        self.caps = np.load('coco/'+dataset+'-captions.npy')
        self.caps = self.caps.reshape((num, self.caps.shape[0]/num, self.caps.shape[-1]))
        if self.caps.shape[-1] != self.lang_N:
            c = np.zeros((num, self.caps.shape[1], self.lang_N))
            c[:,:, :self.caps.shape[-1]] = self.caps
            self.caps = c

        self.images = np.zeros((bs,3,img_size*img_size)).astype('float32')
        self.captions = np.zeros((bs,self.caps.shape[-1])).astype(int)
        self.mask = np.ones((bs,self.caps.shape[-1])).astype(int)


    def reset(self, state):
        self.index = 0
        self.close(state)
        return self.open()    

    def sent2matrix(self, sentence):
        words = sentence.split()
        m = np.int32(np.zeros((1, self.lang_N))) 
        
        for i in xrange(len(words)):
            if words[i] in dictionary:
                m[0,i] = dictionary[words[i]]
            else:
                m[0,i] = dictionary['UNK']
            
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
        mask[...] = 0

        for i in range(bs):
            if self.index >= self.num:
                self.index -= 1
            s = np.random.randint(0,self.caps.shape[1])
                
            # put data in
            captions[i, :] = self.caps[self.index, s, :]
            mask[i, :] = self.caps[self.index, s, :]
            images[i, :] = self.imgs[self.index, :]

            self.index += 1

        # make mask binary
        mask[mask>0] = 1
            
        return (images.reshape((bs, 3*32*32)), captions, mask)
