# Xin Gao, Andrew ID: xing1

import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame
import os,sys
import matplotlib.pyplot as plt
import math

# read and input data
# generate lists for images and labels (train, val)
path = "C:/Users/Xin/Desktop/S17/94867/Dataset/entire"

def import_files(path,ratio):
    df = pd.DataFrame.from_csv("C:/Users/Xin/Desktop/S17/94867/Dataset/Grade.csv")
    d = dict(zip(df.name,df.grade))
    benign = []
    label_benign =[]
    malignant = []
    label_malignant = []

    for filename in os.listdir(path):
        # if not filename.startswith('tr'):
        #     continue
        filename = filename.split(".")[0]
        if filename in d:
            if d[filename]==' benign':
                benign.append(path+'/'+filename)
                label_benign.append(0) # 0 represent benign
            elif d[filename]==' malignant':
                malignant.append(path+'/'+filename)
                label_malignant.append(1) # 1 represent malignant
    print('There are %d benign and %d maglignant tumor images'
           % (len(benign),len(malignant)))


    image_list = np.hstack((benign,malignant))
    label_list = np.hstack((label_benign,label_malignant))
    # random shuffle
    temp = np.array([image_list,label_list]).transpose()
    np.random.shuffle(temp)     
    
    # split the images into subsets - one for training, the other for validation
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    n_sample = len(label_list) # 85+80
    n_val = math.ceil(n_sample*ratio)# validation samples
    n_train = n_sample-n_val
    
    tra_images = image_list[0:n_train]
    tra_labels = label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    
    val_images = image_list[n_train:]
    val_labels = label_list[n_train:]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels,val_images, val_labels

# makes an input queue
def get_batch(image,label,imgH,imgW,batch_size,capacity):
    # convert lists into tensor type
    # capacity: the maximum number of elements in the queue
    image = tf.cast(image,dtype =tf.string)
    label = tf.cast(label,dtype =tf.int32)

    # makes an input queue
    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0] + '.png')
    image = tf.image.decode_png(image_contents) #use tf decoder
    # data preprocessing
    image = tf.image.resize_image_with_crop_or_pad(image,imgH,imgW)
    image = tf.image.per_image_standardization(image) #standadize
    image.set_shape((imgH,imgW,4)) # RGBA
    
    # generate batch
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                      batch_size = batch_size,
                                                      num_threads = 64,
                                                      capacity = capacity,
                                                      min_after_dequeue = capacity -1)

    label_batch = tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch,tf.float32)
    
    return image_batch, label_batch

 
