import tensorflow as tf
from PIL import Image
import input426
import model426
import numpy as np
import random
'''522,775,4'''
  

def run_testing():
    with tf.Graph().as_default(): # define operations and tensors
        BATCH_SIZE = 1
        N_CLASSES = 2
        RATIO = 0.2
        path = "C:/Users/Xin/Desktop/S17/94867/Dataset/test"
        testImage,testLabel,valImage,valLabel = input426.import_files(path,RATIO)
        #oneImage = testImage[1]
        oneImage = random.sample(testImage,1) # randomly select one test sample
        oneImage = "".join(oneImage)+".png" # get the directory of the image
        num = oneImage.split("_")[0][-1]+ oneImage.split("_")[1].split(".")[0]
        
        # convert to ndarray
        im = Image.open(oneImage)
        im.resize((522,775))
        im_array = np.array(im)
        print("test sample %s" % num, "has a shape of",im_array.shape)
                
        image = tf.cast(im_array, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image,522,775)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image,[1,522,775,4])
        
        logit = model426.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape = [522,775,4])
        
        logs_train = "C:/Users/Xin/Desktop/S17/94867/Dataset/log/train"        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            print("loading checkpoint file ...")
            # returns a checkpointState if the state was available
            ckpt = tf.train.get_checkpoint_state(logs_train)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split("-")[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loaded, global step is %s" % global_step)
            else:
                print("File not found")

            pred = sess.run(logit,feed_dict={x:im_array})
            if np.argmax(pred) ==0:
                print('The image is benign with possibility %.4f' % pred[:,0])
            else:
                print('The image is maglinant with possibility %.4f'% pred[:,1])
