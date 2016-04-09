from __future__ import division

import logging
import theano
import theano.tensor as T

import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont
from main_loop import MainLoop
from model import Model
from extensions import Extension

dictionary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digit': 11, 'is': 12, 'on': 13, 'at': 14, 'left': 15, 'right': 16, 'bottom': 17, 'top': 18, 'of': 19, 'image': 20, '.': 21, 'red':22, 'green':23, 'blue':24, 'yellow':25, 'white':26, 'cyan':27, 'magenta':28}
colors = ['red', 'green', 'blue', 'yellow', 'white', 'cyan', 'magenta']

sentences = []
num_sentences=10
masks = np.ones((num_sentences, 14))
for i in range(num_sentences):
    k = np.random.randint(0,7)
    d1 = np.random.randint(0, 9)
    d2 = np.random.randint(0, 9)
    c1 = np.random.randint(0, len(colors)-1)
    c2 = np.random.randint(0, len(colors)-1)
    if k == 0:
        sentence = 'the %s digit %d is on the left of the %s digit %d .' % (colors[c1], d1, colors[c2], d2)
    elif k == 1:
        sentence = 'the %s digit %d is on the right of the %s digit %d .' % (colors[c2], d2, colors[c1], d1)
    elif k == 2:
        sentence = 'the %s digit %d is at the top of the %s digit %d .' % (colors[c1], d1, colors[c2], d2)
    elif k == 3:
        sentence = 'the %s digit %d is at the bottom of the %s digit %d .' % (colors[c2], d2, colors[c1], d1)
    elif k == 4:
        sentence = 'the %s digit %d is at the top left of the image .' % (colors[c1], d1)
        masks[i,-1] = 0
    elif k == 5: 
        sentence = 'the %s digit %d is at the bottom right of the image .' % (colors[c1], d1)
        masks[i,-1] = 0
    elif k == 6:
        sentence = 'the %s digit %d is at the top right of the image .' % (colors[c1], d1)
        masks[i,-1] = 0
    elif k == 7:
        sentence = 'the %s digit %d is at the bottom left of the image .' % (colors[c1], d1)
        massk[i,-1] = 0
        
    sentences.append(sentence)


def sent2matrix(sentence):
    words = sentence.split()
    m = np.int32(np.zeros((1, 14))) 
    
    for i in xrange(len(words)):
        m[0,i] = dictionary[words[i]]
        
    return m

y = np.asarray([sent2matrix(s) for s in sentences]).astype(int)
y = y.reshape((y.shape[0],y.shape[-1]))

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale

logger = logging.getLogger(__name__)

def text(canvas, v,x,y):
    font = ImageFont.truetype('arial.ttf',12)#Vera.ttf', 36)
    canvas.text((x,y), str(v), font=font, fill='black')

class SampleSentences(Extension):

    def __init__(self, save_subdir, bs, img_height, img_width, **kwargs):
        self.iteration = 0
        self.subdir = save_subdir
        self.h = img_height
        self.w = img_width
        self.channels = 3
        self.bs = bs
        global y
        self.y = np.zeros((self.bs, y.shape[1])).astype(int)
        self.y[:y.shape[0],:] = y
        self.mask = np.ones(self.y.shape).astype(int)
        global masks
        self.mask[:y.shape[0],:] = masks

        kwargs.setdefault('before_training', True)
        kwargs.setdefault('after_epoch', True)
        super(SampleSentences, self).__init__(**kwargs)


    def do(self, *args, **kwargs):
        logger.info("Sampling Sentences")
        c = self.main_loop.model.sample_sentences(self.y, self.mask)[0]
        
        img_grid(c, self.h, self.w, self.channels, self.subdir, self.iteration)
        self.iteration += 1
        logger.info("Done Sampling")


def img_grid(imgs, height, width, channels, subdir, iteration, global_scale=True):
    global sentences
    glimpses, bs, _, _ = imgs.shape
    
    imgs = imgs.reshape((glimpses, bs, channels, height, width))
    bs = len(sentences)

    total_height = bs * (height + 20) + 50
    total_width  = glimpses * width + 250
    
    if global_scale:
        imgs = scale_norm(imgs)
        
    I = np.zeros((3, total_height, total_width))
    I.fill(1)

    for i in xrange(bs):
        for j in xrange(glimpses):

            offset_y, offset_x = i*(height+20)+i, j*width+j
            I[:, offset_y:(offset_y+height), offset_x:(offset_x+width)] = imgs[j,i,:,:]

    
    I = (255*I).astype(np.uint8)
    if(I.shape[0] == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I).astype(np.uint8)
        img = Image.fromarray(out)
        canvas = ImageDraw.Draw(img)
        for i in range(bs):

            text(canvas, sentences[i], 10, (i+1)*(height+20)+(i+1)-18)

    img.save("{0}/text2img-{1:04d}.png".format(subdir, iteration))

