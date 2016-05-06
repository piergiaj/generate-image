from __future__ import division
import cPickle

import logging
import theano
import theano.tensor as T

import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont
from main_loop import MainLoop
from model import Model
from extensions import Extension


dictionary = cPickle.load(open('coco/dictionary.pkl','rb'))

sentences = ['a stop sign is flying in blue skies .', 'a herd of elephants flying in blue skies .', 'a toilet seat sits open in a grass field .', 'a person skiing on sand clad vast desert .', 'a yellow school bus parked in a parking lot .', 'a green school bus parked in a parking lot .', 'a red school bus parked in a parking lot .', 'a blue school bus parked in a parking lot .', 'the chocolate desert is on the table .', 'a bowl of bananas is on the table .', 'a vintage photo of a cat .', 'a vintage photo of a dog .', 'a very large commercial plane flying in blue skies .', 'a very large commercial plane flying in rainy skies .', 'a herd of elephants walking across a dry grass field .', 'a herd of elephants walking across a green grass field .', 'a rider on a motorcycle in the desert .', 'a rider on a motorcycle in the forest .', 'a surfer woman and child walk on the beach .', 'a surfer woman and child walk on the sun .']
#sentences = ['a wooden bench sitting in the middle of a forest .', 'an old silver and brown double parking meter', 'a man seated at a table in a restaurant , with a pizza in front of him that has various toppings .', 'a herb of wild zebra feed on grass', 'a living room with a couch and a tv near a wall', 'a cat wearing a tie , laying on a large soft surface .', 'a photo through a window looking a planes parked at the terminal .', 'a group of people walking around a lush green field .', 'a woman is sitting at a table and eating with two children .', 'a woman throwing a tennis ball in the air and a racket it her other hand .', 'a bathroom with a toilet , shower , and sink', 'a hand strokes the face of a man playing a video game .', 'older cat is sitting on a toilet inside of the bathroom .', 'an airplane in the sky with all the landing gear deployed .', 'a plate with a clown face made from sliced vegetables on sliced cheese .', 'the surfer is taking air while surfing a wave .', 'a book atop the arm of a bench .', 'a black cup with a spoon sticking out next to a folded pair of glasses .', 'he is ready to block the flight of that frisbee .', 'on a dark street a red fire hydrant glows .', 'man standing in open door of car on a desert road .', 'a woman holding shears in one hand a tile in the other .', 'two birds sitting on the the back of a giraffe .', 'a male emo hipster wearing a furry jacket in front of a laptop computer .', 'a stop sign at an intersection with trees and a building behind it .', 'a dog that is standing near an open window .', 'a young boy eats something in front of a bike', 'a man on pier with dog jumping for frisbee into water .', 'there are two people enjoying a wedding reception', 'four green chairs attached to wall with white clock above', 'a personal pizza , grapes , and a beverage sit on a tray table .', 'a dog tied up to a metal cage next to a back pack .', 'a sandwich and salad and fries with a dip', 'lit up night traffic is zooming by a clock tower .', 'a woman and a dog sit in a huge outdoor chair .', 'man on ocean beach flying several kites on windy day .', 'a woman holding a cell phone while a purse is over her shoulder .', 'a person walking with a plastic bag', 'a train sitting in a train station in a rural area .', 'a group of people gathered at the bottom of a snow mountain', 'a person is riding on a jet ski in a body of water .', 'an animal standing on top of a lush green field .', 'a hot dog with mayo and other toppings on a hoagie bun .', 'a large modern hotel room with double beds', 'the christmas figurines are each playing different instruments .', 'a bathroom toilet with a toilet paper roll and seat cover dispenser .', 'a dog that is on a black skateboard', 'a young woman standing next to a pink frisbee .', 'a black motorcycle parked in front of trees .', 'the airplane is on the ground , but getting ready for takeoff .', 'two women eat chili dogs on a city sidewalk .', 'a woman looking at laptop as she sits at a table', 'a very nice large modern style kitchen with a bar .', 'a seagull bends its head backwards to preen its back feathers', 'a parking meter with a bike and a car next to it .', 'a sandwich and salad and fries with a dip', 'a black cat walks gingerly around an empty wine bottle .', 'a bunch of zebras are standing in a field', 'an airplane sitting at an airport while a man looks inside of it .', 'a tractor trailer is parked on the side of a country road .', 'the clock is on display on the side of the building outside .', 'first baseman throwing a ball on a field', 'a birthday cake with candy on top of it', 'black and white photo of a stop sign on a rural street .', 'a UNK farm truck in a mountain field beside trees .', 'a dog on a leash attached to a wooden bench .', 'a white plate with crumbled chocolate cake on it .', 'a man is holding a plate with breakfast food .', 'a couple of airplanes that are parked on the runway', 'a giraffe walking through a tree filled forest .', 'a table is laid out with several different doughnuts and pastries .', 'a green traffic light above a street with a car on coming .', 'a clock tower on top of a church with a weather vein .', 'a piece of partially-eaten cake sits on a paper plate .', 'a man is playing a video game in the living room', 'an elephant walks alone past some big rocks boulders in an open field', 'a bathroom scene with a toilet and a sink .', 'a bull with very large horns and a calf .', 'two teams chase a player that has kicked a soccer ball .', 'commuter bus on roadway in large city with traffic .', 'the black dog sits on the wooden floor watching a television .', 'a wooden park bench next to a country road .', 'man lying down on bed with shirt open in bedroom', 'two suitcases made of leather and stacked on top of each other', 'two giraffes standing neck to neck near some brush .', 'a train traveling on top of a bridge spanning a river .', 'there is a big horn sheep standing on the rocky side of a mountain .', 'a boat named after author UNK UNK with a dragon painted on it', 'the plane is taking off over the trees .', 'a couple of beds that are in one room', 'a dog looking out over sailboats at a marina .', 'a dog lies on the carpet with a stuffed animal on its head .', 'many buses drive down a street in a single file line', 'two-tiered white fondant cake on pedestal with tiers separated by two rows of white and green roses covered with a dark wire UNK birdcage top with a bird perched atop for a handle .', 'a calico cat faces the camera while wearing a tie .', 'a close up of a cake in the shape of a truck', 'a crowded beach with many kites flying', 'a few people are getting of a plane .', 'a large french bread pizza spanning the length of a table .', 'a gigantic teddy bear sprawled out on the couch .', 'a man on a surfboard riding a wave .', 'a very long submarine sandwich sitting on a table next to bottles of alcohol .', 'several signs attached to a post detail UNK related to parking and other issues .', 'a living room in a remotely located home .', 'many people on a courtyard under a clock .', 'a man standing on a tennis court near a large net .', 'a person standing in a field flying a kite .', 'a purple and white bus in a parking lot .', 'a commercial jet liner being taxied on runway .', 'a bathroom that features a vanity cabinet with sink , commode , overhead cabinet and mirror .']

def sent2matrix(sentence):
    words = sentence.split()
    m = np.int32(np.zeros((1, 57))) 
    
    for i in xrange(len(words)):
        if words[i] in dictionary:
            m[0,i] = dictionary[words[i]]
        else:
            m[0,i] = dictionary['UNK']
        
    return m

# 5 duplicates of each sentence
y = np.asarray([[sent2matrix(s) for j in range(5)] for s in sentences]).astype(int).reshape((100,57))

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
        if bs > y.shape[0]:
            self.y[:y.shape[0],:] = y
        else:
            self.y = y[:bs]
        self.mask = np.ones(self.y.shape).astype(int)

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
    bs = 10#len(sentences)

    total_height = bs * (height + 20) + 50
    total_width  = glimpses * width + 250
    
    if global_scale:
        imgs = scale_norm(imgs)
        
    I = np.zeros((3, total_height, total_width))
    I.fill(1)

    si = np.random.randint(0,len(sentences),(bs,))

    # draw each of the 5 generated images
    for i in xrange(bs):
        for j in xrange(5):
            offset_y, offset_x = i*(height+20)+i, j*width+j
            I[:, offset_y:(offset_y+height), offset_x:(offset_x+width)] = imgs[0,si[i]*5+j,:,:]

    
    I = (255*I).astype(np.uint8)
    if(I.shape[0] == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I).astype(np.uint8)
        img = Image.fromarray(out)
        canvas = ImageDraw.Draw(img)
        for i in range(bs):

            text(canvas, sentences[si[i]], 10, (i+1)*(height+20)+(i+1)-18)

    img.save("{0}/text2img-{1:04d}.png".format(subdir, iteration))

