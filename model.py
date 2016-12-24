import numpy as np
import csv
import pandas as pd
import scipy.misc as sp
import json
import cv2 # for warpAffine

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.models import Sequential, model_from_json
from keras.layers import Input, Conv2D, Activation, Dropout, Flatten, Dense, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Set tensorflow session to not use all GPU memory
GPU_fraction_to_use = 0.8
def get_session(gpu_fraction=GPU_fraction_to_use):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())




# read the driving log for image names and steering angles
reader = csv.reader(open('driving_log.csv'), delimiter=',')
cimage_names = []
steering = []
first_line = True
for row in reader:
    #print(row)
    if first_line:
        first_line = False
        continue
    cimage_names.append(row[0].strip())
    steering.append(float(row[3]))
steering = np.array(steering)
print ('number of images in original training set: {}'.format(len(cimage_names)))

# Load center images into one array in memory, resizing them to (80, 160).
# I managed to use in-memory array of the original images as my laptop can handle it,
# but I still do use generator to generate extra augmented images for 'recovery'
interp = 'cubic'
resize_shape = (80,160)
cimages = np.stack( [sp.imresize(sp.imread(file, mode='RGB'), size=resize_shape, interp=interp) 
                      for file in cimage_names], 
                    axis=0 )
print('shape of the in-memory images array is: {}'.format(cimages.shape))

# Creating a mask to black out top of the image and bottom of the image.
# Top features are irrelevant for steering. Without them model should generalise better.
# Bottom of the images have car bonnet which will be unhelpful feature in augmented images,
# so we do not want the model to learn it.
# What is left is essentially view of the road to be learned from.
shape = (None,)+resize_shape+(3,)
mask = np.zeros(shape[1:])
mask[0:27,:,:] = 0
mask[27:65,:,:] = 1
mask[65:,:,:] = 0
print('masking array shape is: {}'.format(mask.shape))
# APPLY MASK to the center images array
cimages = np.multiply(cimages,mask).astype(np.uint8)




# Helper to be used in the generator to generate augmented images.
# Takes and image and steering angle and applies random affine transform, essentially shifting
# the original image up/down and left/right.
# Trans_range is parameter in pixels to specify the range by how much to shift.
# Applies transformation to the steering angle as well, which allows model to train for 'recovery'
# Credit: Vivek Yadav who published it here: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.44iyh6lz0
def trans_image(image, steer, trans_range):
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(shape[2], shape[1]))
    return image_tr,steer_ang

# Training data generator.
# batch_size is the size of the mini-batch to be fed into one forward/backward pass of the optimizer.
# Shuffles the input set for each epoch.
# Generates images in threes: one random original image+steering, 
#                             another random images/steering mirrorred horizontally
#                             first random original image+steering transformed randomly for recovery
def train_Generator(input_images, input_steering, batch_size=64):
    n = len(input_images)
    shape = (batch_size,)+input_images.shape[1:]
    # pre-allocate arrays for efficiency
    batch_images = np.zeros(shape, dtype=np.uint8)
    batch_steering = np.zeros(batch_size)
    # infinite generator
    while 1:
        # shuffle the array of center images randomly
        perm = np.random.permutation(range(n))
        current_batch_cnt = 0
        for i in perm: 
            # feed one original center image from random location
            batch_images[current_batch_cnt,:,:,:] = input_images[i]
            batch_steering[current_batch_cnt] = input_steering[i]
            current_batch_cnt += 1
            # feed one original center image mirrored horizontally. from another part of the set
            j = perm[n-1-i]
            batch_images[current_batch_cnt,:,:,:] = input_images[j][:,::-1,:]
            batch_steering[current_batch_cnt] = -input_steering[j]
            current_batch_cnt += 1
            # feed same center image as previously, but augmented randomly. to train for 'recovery'
            trrange = 20
            im, st = trans_image(input_images[i], steer=input_steering[i], trans_range=trrange)
            batch_images[current_batch_cnt,:,:,:] = im
            batch_steering[current_batch_cnt] = st
            current_batch_cnt += 1

            # when batch is full size, feed it
            if current_batch_cnt == batch_size or current_batch_cnt+3>batch_size:
                yield batch_images, batch_steering
                current_batch_cnt = 0



# define the CNN.
# Architecture inspired by NVIDIA paper: 
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def createModel(input_shape):
    model = Sequential()
    # normalize
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape))
    # convolutions
    model.add(Conv2D(24, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3, border_mode='valid'))
    # dropout
    model.add((Dropout(0.1)))
    model.add(Activation('relu'))
    # fully connected layers
    model.add(Flatten()) 
    model.add(Dense(300, activation='relu'))
    model.add(Dense(10))
    # output steering
    model.add(Dense(1))
    return model

# create model
model = createModel(cimages.shape[1:])
model.summary()
# define learning parameters and optimizer
learning_rate= 0.001 # using the default initial learning rate with Adam optimizer
batch_size = 64
epochs = 5
optimizer = Adam(lr=learning_rate)

# compile
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
# save the model definition
json_str = model.to_json()
with open('model.json','w') as f:
    json.dump(json_str, f)


# NB I do NOT use train/test/validation split as in this problem.
# It turns out that the only true
# test is if the car can drive the track on its own.
# I have found little correlation between the MSE of the trained model on train set
# and the actual performance on the track. And also plotting the predictions vs the inputs
# does not instill confidence, but it still drives perfectly!


# Train the model using generator and requied number of epochs
# Takes about 60 seconds per epoch on my Macbook Pro 2013, NVidia 750M/2GB
history = model.fit_generator(train_Generator(cimages, steering, batch_size), 
                              samples_per_epoch=len(cimage_names)*3, # this is because for each original image generator returns 3 
                              nb_epoch=epochs,
                              verbose=1)
# save trained model weights
model.save_weights('model.h5')


