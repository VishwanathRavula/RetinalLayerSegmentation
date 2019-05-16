#!/usr/bin/env python
# coding: utf-8
# [0.99011564 0.92536699 0.93408108 0.91316754 0.85130454 0.97386465 0.94672452 0.90012799]

import keras
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Deconvolution2D
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Lambda,Add
from keras.utils import to_categorical
import tensorflow as tf

from keras.layers import Reshape

from keras import backend as K, Sequential
from keras import regularizers, optimizers
#get_ipython().magic(u'matplotlib inline')

from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint


import scipy.io as scio
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import re
from scipy.misc import imsave
from scipy import ndimage, misc
from numpy import unravel_index
from operator import sub

import os
cwd = os.getcwd()

#Function to convert string to int
def atoi(text) :
    return int(text) if text.isdigit() else text


# Function used to specify the key for sorting filenames in the directory
# Split the input based on presence of digits
def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]

# Sorting the files in the directory specified based on filenames
root_path = ""
filenames = []
for root, dirnames, filenames in os.walk("DenoisedTrain"):
    filenames.sort(key = natural_keys)
    rootpath = root
# print('Filenames: ',filenames)

# Reads the images as per the sorted filenames and stores it in a list
images = []
for filename in filenames :
    filepath = os.path.join(root,filename)
    image = ndimage.imread(filepath, mode = "L")
    images.append(image)
    print(filename)

print('Total Images: ',len(images))

#Loading the labels for resized cropped images
labels = np.load('resized_cropped_labeledimages.npy')
labels_list = []
for i in range(len(labels)):
    labels_list.append(labels[i])
print(labels.shape)


# In[11]:


train_labels = np.zeros((770,216,64,8))


#Loop to perform one-hot encoding for the labels
for i in range(len(labels_list)) :
    for j in range(216) :
        for k in range(64):
            if(labels_list[i][j][k] == 0):
                train_labels[i][j][k][0] = 1
            if(labels_list[i][j][k] == 1):
                train_labels[i][j][k][1] = 1
            if(labels_list[i][j][k] == 2):
                train_labels[i][j][k][2] = 1
            if(labels_list[i][j][k] == 3):
                train_labels[i][j][k][3] = 1
            if(labels_list[i][j][k] == 4):
                train_labels[i][j][k][4] = 1
            if(labels_list[i][j][k] == 5):
                train_labels[i][j][k][5] = 1
            if(labels_list[i][j][k] == 6):
                train_labels[i][j][k][6] = 1
            if(labels_list[i][j][k] == 7):
                train_labels[i][j][k][7] = 1

images=np.array(images)
print(images.shape)
images = images.reshape(images.shape[0],216,64,1)

print(images.shape)
#Generate a random train set from 770 indices
train_indices = np.random.choice(770,500,replace = False)
print(sorted(train_indices))
train_images_random = []
train_labels_random = []

#Create the train set (images and labels) based on the randomly generated train indices
for i in train_indices:
    train_images_random.append(images[i])
    train_labels_random.append(train_labels[i])

#Generate the test set from the original image list by excluding the train indices

test_indices = [x for x in range(770) if x not in train_indices]
print(test_indices)
test_images = []
test_labels = []
for i in test_indices:
    test_images.append(images[i])
    test_labels.append(train_labels[i])


train_images = np.array(train_images_random)
train_labels = np.array(train_labels_random)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Convert the train and test lists (train images, train labels, test images, test labels) to numpy arrays of type float32
train_images = train_images.astype('float32')
train_labels = train_labels.astype('float32')
test_images = test_images.astype('float32')
test_labels = test_labels.astype('float32')


#Set the input image shape to 216x500
data_shape = 216*64
#Set the weight decay parameter for Frobenius norm to 0.001
weight_decay = 0.0001

""" Model definition for retinal layer segmentation using dilated convolutions"""

"""
Each encoder block consists of Convolutional layer, Batch Normalization layer, ReLU activation and a Max pooling layer
Each dilation layer consists of a dilated convolutional filter with Batch Normalization layer and ReLU activation
Each decoder block consists of Convolutional layer, Batch Normalization layer, ReLU activation and Upsampling layer 
Additive skip connections transfer the features from encoder to the corresponding decoder blocks respectively
Classification path consists of a convolutional layer of kernel size 1x1 and a Softmax activation function
"""

big_filter = False
inputs = Input(shape=(216,64,1))
if big_filter :
    # Defines the input tensor

    L1 = Conv2D(64,kernel_size=(7,3),activation = 'relu',padding = "same")(inputs)
    L2 = BatchNormalization()(L1)
    #L3 = Lambda(maxpool_1,output_shape = shape)(L2)
    L3 = MaxPooling2D(pool_size=(2,2))(L2)
    L4 = Conv2D(64,kernel_size=(7,3),activation = 'relu',padding = "same")(L3)
    L5 = BatchNormalization()(L4)

    #L6 = Lambda(maxpool_2,output_shape = shape)(L5)
    L6 = MaxPooling2D(pool_size=(2,2))(L5)
    L7 = Conv2D(64,kernel_size=(7,3),activation = 'relu',padding = "same")(L6)
    L8 = BatchNormalization()(L7)
    #L9 = Lambda(maxpool_3,output_shape = shape)(L8)
    L9 = MaxPooling2D(pool_size=(2,2))(L8)
    L10 = Conv2D(64,kernel_size=(7,3),activation = 'relu',padding = "same")(L9)
    L11 = BatchNormalization()(L10)
    L12 = UpSampling2D(size = (2,2))(L11)
    #L12 = Lambda(unpool_3,output_shape = unpool_shape)(L11)
    L13 = Concatenate(axis = 3)([L8,L12])
    L14 = Conv2D(64,kernel_size=(7,3),activation = 'relu',padding = "same")(L13)
    L15 = BatchNormalization()(L14)
    L16 = UpSampling2D(size= (2,2))(L15)
    #L16 = Lambda(unpool_2,output_shape=unpool_shape)(L15)
    L17 = Concatenate(axis = 3)([L16,L5])
    L18 = Conv2D(64,kernel_size=(7,3),activation = 'relu',padding = "same")(L17)
    L19 = BatchNormalization()(L18)
    #L20 = Lambda(unpool_1,output_shape=unpool_shape)(L19)
    L20 = UpSampling2D(size=(2,2),name = "Layer19")(L19)
    L21 = Concatenate(axis=3)([L20,L2])
    L22 = Conv2D(64,kernel_size=(7,3),activation = 'relu',padding = "same")(L21)
    L23 = BatchNormalization()(L22)
    L24 = Conv2D(8,kernel_size=(1,1),padding = "same")(L23)
    L25 = Reshape((data_shape,8),input_shape = (216,64,8))(L24)
    L26 = Activation('softmax')(L25)


    model = Model(inputs = inputs, outputs = L26)
    model.summary()
else :
     # Defines the input tensor

    L1 = Conv2D(64,kernel_size=(5,3),activation = 'relu',padding = "same")(inputs)
    L2 = BatchNormalization()(L1)
    #L3 = Lambda(maxpool_1,output_shape = shape)(L2)
    L3 = MaxPooling2D(pool_size=(2,2))(L2)
    L4 = Conv2D(64,kernel_size=(5,3),activation = 'relu',padding = "same")(L3)
    L5 = BatchNormalization()(L4)

    #L6 = Lambda(maxpool_2,output_shape = shape)(L5)
    L6 = MaxPooling2D(pool_size=(2,2))(L5)
    L7 = Conv2D(64,kernel_size=(5,3),activation = 'relu',padding = "same")(L6)
    L8 = BatchNormalization()(L7)
    #L9 = Lambda(maxpool_3,output_shape = shape)(L8)
    L9 = MaxPooling2D(pool_size=(2,2))(L8)
    L10 = Conv2D(64,kernel_size=(5,3),activation = 'relu',padding = "same")(L9)
    L11 = BatchNormalization()(L10)
    L12 = UpSampling2D(size = (2,2))(L11)
    #L12 = Lambda(unpool_3,output_shape = unpool_shape)(L11)
    L13 = Concatenate(axis = 3)([L8,L12])
    L14 = Conv2D(64,kernel_size=(5,3),activation = 'relu',padding = "same")(L13)
    L15 = BatchNormalization()(L14)
    L16 = UpSampling2D(size= (2,2))(L15)
    #L16 = Lambda(unpool_2,output_shape=unpool_shape)(L15)
    L17 = Concatenate(axis = 3)([L16,L5])
    L18 = Conv2D(64,kernel_size=(5,3),activation = 'relu',padding = "same")(L17)
    L19 = BatchNormalization()(L18)
    #L20 = Lambda(unpool_1,output_shape=unpool_shape)(L19)
    L20 = UpSampling2D(size=(2,2),name = "Layer19")(L19)
    L21 = Concatenate(axis=3)([L20,L2])
    L22 = Conv2D(64,kernel_size=(5,3),activation = 'relu',padding = "same")(L21)
    L23 = BatchNormalization()(L22)
    L24 = Conv2D(8,kernel_size=(1,1),padding = "same")(L23)
    L25 = Reshape((data_shape,8),input_shape = (216,64,8))(L24)
    L26 = Activation('softmax')(L25)


    model = Model(inputs = inputs, outputs = L26)
    model.summary()

"""End of model defination"""

# Load the pre-trained weights if already trained
# Already trained weights are available in Model_weights/
# model.load_weights("RelaynetO_5.hdf5")

# from keras.utils import plot_model
# plot_model(model, to_file='model2_add_up.png',show_shapes= True)
# Load the weighted images obtained after pre-processing
weights = np.load('weighted_cropped_images.npy')

weights.shape

np.unique(weights)

weights_matrix = []
for i in train_indices:
    weights_matrix.append(weights[i])

sample_weights = np.array(weights_matrix)

sample_weights = np.reshape(sample_weights, (500, data_shape))

# Smoothing parameter for computation of dice co-efficient
train_labels = np.reshape(train_labels, (500, data_shape, 8))
test_labels = np.reshape(test_labels, (270, data_shape, 8))
smooth = 1

# Calculation of the dice co-efficient based on actual and predicted labels
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Dice loss computed as -dice co-efficient
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Combined loss of weighted multi-class logistic loss and dice loss
def customized_loss(y_true, y_pred):
    return (1 * K.categorical_crossentropy(y_true, y_pred)) + (0.5 * dice_coef_loss(y_true, y_pred))

# Using SGD optimiser with Nesterov momentum and a learning rate of 0.001
optimiser = optimizers.SGD(lr=0.005, momentum=0.9, nesterov=True)
# optimiser = 'Adam'

# Compiling the model
model.compile(optimizer=optimiser, loss=customized_loss, metrics=['accuracy', dice_coef], sample_weight_mode='temporal')

# Defining Callback functions which will be called by model during runtime when specified condition is satisfied
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
csv_logger = CSVLogger('exp1.csv')
model_checkpoint = ModelCheckpoint("exp1.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

"""Train the model by specifying all the required arguments
Sample weights are passed to the sample_weight argument - Numpy array of weights for the training samples, 
used for weighting the loss function (during training only)
"""

class_weighting = [1.0,80.0000,80.00000000,80.00000000,80.0000,80.000,80.00,80.00]

model.fit(train_images, train_labels, batch_size=32, epochs=100, validation_data=(test_images, test_labels),
          sample_weight=sample_weights, callbacks=[lr_reducer, csv_logger, model_checkpoint],class_weight=class_weighting)

def test_preprocessing(test_image):
    test_image = np.squeeze(test_image, axis=2)
    test_image = test_image.reshape((1, 216, 64, 1))
    return test_image

# Computation of the layer-wise dice scores
def dice_score(layer, y_true, y_pred):
    y_true_layer = y_true[:, layer]
    y_pred_layer = y_pred[:, layer]
    # print(y_true_layer.shape)
    # print(y_pred_layer.shape)
    intersection = np.dot(y_true_layer, y_pred_layer.T)
    score = 2. * intersection / (np.sum(y_true_layer) + np.sum(y_pred_layer))
    return score

test_images_size = len(test_images)
test_images_size

# Run the test set on the trained model and compute the layer wise dice scores for each test image
test_dice_scores = np.zeros((test_images_size, 8))
for image_no in range(test_images_size):
    prediction = model.predict(test_preprocessing(test_images[image_no]))
    # print(prediction.shape)
    prediction = np.squeeze(prediction, axis=0)
    # print(prediction.shape)
    print(image_no)
    for layer_no in range(8):
        test_dice_scores[image_no][layer_no] = dice_score(layer_no, test_labels[image_no], prediction)

# Compute the mean dice score over all images for each of the retinal layers
overall_dice_scores = np.zeros((8))
for layer_no in range(8):
    overall_dice_scores[layer_no] = np.mean(test_dice_scores[:, layer_no])

print(overall_dice_scores)

