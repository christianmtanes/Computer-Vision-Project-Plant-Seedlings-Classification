# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:18:13 2018

@author: Chris
"""


"""
we will import two classes from keras
"""
"""
First is sequential this means we’re going to have a model that’s a sequence
of layers one after the other.
"""
from tensorflow.python.keras.models import Sequential
"""
Second we would want to add  a dense layer to the sequential model so imported that.
"""
from tensorflow.python.keras.layers import Dense


from tensorflow.python.keras.applications import ResNet50

from tensorflow.python import keras

import tensorflow as tf
import numpy as np
import pandas as pd


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

"""num classes will be equal to the number of categories we want to classify"""
num_classes = 12

"""
size of images is the default size that it was trained on ImageNet dataset
"""
image_size = 224
target_size = (image_size, image_size) 



"""
we set up a sequential model that we can add layers to
"""
my_new_model = Sequential()
"""
first we add all of pre-trained ResNet 50 model
we've written include_top=False, this is how specify that we want to exlude
the layer that makes prediction into the thousands of categories used in the ImageNet competition
we set the weights to be 'ImageNet' to specify that we use the pre-traind model on ImageNet
pooling equals average says that if we had extra channels in our tensor at the end of this step
we want to collapse them to 1d tensor by taking an average across channels
now we have a pre-trained model that creates the layer before the last layer
that we saw in the slides
"""
my_new_model.add(ResNet50(weights='imagenet', include_top=False, pooling='avg'))
"""
we add a dense layer to make predictions,
we specify the number of nodes in this layer which in this case is
the number of classes,
then we want to apply the softmax function to turn it into probabilities 
"""
my_new_model.add(Dense(num_classes,activation='softmax',))

"""
we tell tensor flow not to train the first layer which is the ResNet50 model
because that's the model that was already pre-trained with the ImageNet data
"""
my_new_model.layers[0].trainable = False
"""
the compile command tells tensorflow how to update the relationships in the dense connections 
when we're doing the training with our data, we have a measure of loss or inaccuracy we want 
to minimize we specify as categorical cross entropy (log loss function) 
we use The Adam optimization algorithm which is an extension to stochastic gradient descent to
minimize the categorical cross entropy, we ask it to report the accuracy metric
that is what fraction of predictions were correct this is easier to interpret than categorical cross entropy scores,
so it would prints out how the model is doing
"""
my_new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
my_new_model.summary()

"""
our data is broken to a directory of train data
and a directory of validation data
within each of thos we have 12 subdirectories (classes) of plant seedlings images
"""
"""

"""
from tensorflow.python.keras.applications.resnet50 import preprocess_input
#from tensorflow.python.keras.applications.inception_v3 import preprocess_input
#from tensorflow.python.keras.applications.vgg16 import preprocess_input
"""
keras provides a tool for working with images grouped into directories
by their label this is the ImageDataGenerator 
"""
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

"""
we create the generator object in the abstract 
telling it that we want to apply the ResNet pre-processing
function every time it reads an image, we use this function to be consistent with how
the resnet model is created
"""
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


"""
we use the flow from directory command, we tell it what directory the data is in, 
target_size = what size image we want,
batch size = how many images to read in at a time,
and that we tell it we're classifying data into different categories(class_mode='categorical')
more information about the choice of batch size in the slides
"""
train_generator = data_generator.flow_from_directory(
        'train',
        target_size=target_size,
       
        class_mode='categorical')
"""
we do the as above to setup the way to read the validation data
that creates a validtion generator
"""
validation_generator = data_generator.flow_from_directory(
        'valid',
        target_size=target_size,
        
        class_mode='categorical')
"""
the ImageDataGenerator is especially valuable when working with
large data sets because we don't need to hold the whole data set
in memory at once 
"""

"""
now we fit the model
we tell it the training data comes from train_generator
we are set to read 19 images at a time we have 3800 images 
so we go 200 steps (steps_per_epoch) of 19 images,
the validation data comes from validation generator
the validation generator reads 5 image at a time and we
have 950 images so the validation steps is 190
"""
my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        validation_data=validation_generator,
        validation_steps=1
        ,epochs=3)

"""
as the model training is running we'll see progress updates showing
with our loss function and the accuracy it updates the connections
in the dense layer and it makes those updates in 200 steps

"""
"""
using the model on test data
"""
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

def predict(model, img):
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    prediction_index = np.argmax(preds[0])
    return CATEGORIES[prediction_index]

submission = pd.read_csv('sample_submission.csv')
for index, row in submission.iterrows():
    filename = 'test/' + row['file']
    img = load_img(filename, target_size=target_size)
    row['species'] = predict(my_new_model, img)
    
submission.to_csv('resnet50_batch_size_32_32.csv', index=False)
