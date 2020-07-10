#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
import sys, time, os, warnings 
import numpy as np
import pandas as pd 
from collections import Counter 
warnings.filterwarnings("ignore")

# print("python {}".format(sys.version))
# print("keras version {}".format(keras.__version__)); del keras
# print("tensorflow version {}".format(tf.__version__))
# config = tf.compat.v1.ConfigProto()
# # # config.gpu_options.per_process_gpu_memory_fraction = 0.95
# # # config.gpu_options.visible_device_list = "0"
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# def set_seed(sd=123):
#     from numpy.random import seed
#     from tensorflow import set_random_seed
#     import random as rn
#     ## numpy random seed
#     seed(sd)
#     ## core python's random number 
#     rn.seed(sd)
#     ## tensor flow's random number
#     set_random_seed(sd)


from keras.applications import VGG16
from keras import models

modelvgg = VGG16(include_top=True)
modelvgg.layers.pop() #pop doesn't work here, although works in google colab
modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-2].output)



from keras.models import load_model
# from keras.models import model_from_json
from tensorflow.keras.models import model_from_json


# load json and create model
json_file = open('imageCaption_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("imageCaption_model.h5")
print("Loaded model from disk")


from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from collections import OrderedDict

images = OrderedDict()
npix = 224
target_size = (npix,npix,3) 


# with the help of this model we are predicting or extracting
# the features of the images which we then store in images dict
def encode_image(filename):
    # load an image from file
    image = load_img(filename, target_size=target_size)
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    nimage = preprocess_input(image)
    print(nimage.shape)
    print(nimage.reshape( (1,) + nimage.shape[:3]).shape)
    feature_vector = modelvgg.predict(nimage.reshape( (1,) + nimage.shape[:3]))
    print(feature_vector.shape)
#     print(feature_vector.shape.flatten())
    
    return feature_vector.flatten()



#loading our "tokenizer" or "word_to_index" dict and "index_to_word" dict
import pickle

with open("word_to_idx.pkl","rb") as w2i:
    tokenizer = pickle.load(w2i)

with open("idx_to_word.pkl","rb") as i2w:
    index_to_word = pickle.load(i2w)



# below function is for predicting caption for our image
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def predict_caption(img):
    # image.shape = (1,4462)
    # the input should be a vector of above size
        
    maxlen = 30    # this may change depending upon preprocessing every time you run the prgm
    in_text = 'startseq'

    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0] # it is a list in list hence we are taking [0]
        sequence = pad_sequences([sequence],maxlen) 
        # since our caption's max length is 33, we are creating a sequence of length 33 for every caption
        yhat = loaded_model.predict([img,sequence],verbose=0)
        yhat = np.argmax(yhat) # gives the index of the max value
        newword = index_to_word[yhat] # we get our next word here
        in_text += " " + newword

        if newword == "endseq":
            break

    return in_text



def caption_this(filename):
  image_load = load_img(filename, target_size=(224,224,3))

  # convert the image pixels to a numpy array
  image = img_to_array(image_load)
  nimage = preprocess_input(image)
  y_pred = modelvgg.predict(nimage.reshape( (1,) + nimage.shape[:3]))
  image_feature = y_pred.flatten()
  caption = predict_caption(image_feature.reshape(1,len(image_feature)))
  caption = caption.split()
  caption = " ".join([word for word in caption if word not in ["startseq","endseq"]])
  # fig = plt.figure(figsize=(7,10))
  # ax = fig.add_subplot(2,1,1,xticks=[],yticks=[])
  # ax.imshow(image_load)
  # ax = fig.add_subplot(2,1,2)
  # plt.axis('off')
  # ax.plot()
  # ax.set_xlim(0,1)
  # ax.set_ylim(0,1)
  # ax.text(0.2,1.1,caption,fontsize=13)

  # plt.show()
  return caption



# caption_this('./static/surf.jpg')





