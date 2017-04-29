# Xin Gao, Andrew ID: xing1

import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame
import os,sys
import math

# read and input data
# generate lists for images and labels (train, val)

#path = "C:/Users/Xin/Desktop/S17/94867/Dataset/train"

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
    print('There are %d benign and %d maglignant images'% (len(benign),len(malignant)))

    imgList = np.hstack((benign,malignant))
    labList = np.hstack((label_benign,label_malignant))
    # random shuffle
    temp = np.array([imgList,labList]).transpose()
    np.random.shuffle(temp)     
    
    # split the images into subsets - one for training, the other for validation
    imgList = list(temp[:,0])
    
    labList = list(temp[:,1])
    nTotal = len(labList) # = 85
    nVal = math.ceil(nTotal*ratio)# num of validation samples
    nTrain = nTotal-nVal # num of training samples
    
    traIMG = imgList[0:nTrain]
    traLBL = labList[0:nTrain]
    traLBL = [int(float(i)) for i in traLBL]
    
    valIMG = imgList[nTrain:]
    valLBL = labList[nTrain:]
    valLBL = [int(float(i)) for i in valLBL]

    return traIMG, traLBL,valIMG, valLBL

# ratio = 0.2
# print(import_files(path,ratio))

# create an input queue
def get_batch(image,label,imgH,imgW,batch_size,capacity):
    # convert lists into tensor type
    image = tf.cast(image,dtype =tf.string)
    label = tf.cast(label,dtype =tf.int32)

    # produce a slice of each Tensor in the tensor_list
    inputQ = tf.train.slice_input_producer([image,label])

    label = inputQ[1]
    img_contents = tf.read_file(inputQ[0] + '.png')
    image = tf.image.decode_png(img_contents) #use tf decoder
    # data preprocessing
    image = tf.image.resize_image_with_crop_or_pad(image,imgH,imgW)
    image = tf.image.per_image_standardization(image) #standadize
    image.set_shape((imgH,imgW,4)) # RGBA
    
    # creates batches by randomly shuffling tensors
    img_batch,lbl_batch = tf.train.shuffle_batch([image,label],
                                                batch_size = batch_size,
                                                num_threads = 64,
                                                capacity = capacity,
                                                min_after_dequeue = capacity -1)

    lbl_batch = tf.reshape(lbl_batch,[batch_size])
    img_batch = tf.cast(img_batch,tf.float32) # required by following training
    
    # data augmentation
    #image = tf.image.random_flip_left_right(image) # flip an image horizontally
    #image = tf.image.random_flip_up_down(image) # flip an image vertically
    
    return img_batch, lbl_batch

 
