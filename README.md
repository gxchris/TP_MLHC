Image Classification for Colorectal Cancer Diagnosis 
 
This project aims to use CNN to do image classification and enable users to predict a label (benign or malignant) for a given
HE-colored colon image

Getting Started - 
Prerequisites

Install python (including numpy and pandas) and tensorflow:

Python with related packages: Download python3.5 (only this version is compatible with latest version of tensorflow) and 
use Command Prompt to install required packages. The most effective way to install ml packages is to 
download anaconda which includes more than 620 analytic packages

Tensorflow: refer to https://www.tensorflow.org/install/ for detailed instructions. 
If you have anaconda installed, you could refer to https://anaconda.org/jjhelmus/tensorflow to pip install this package

Enter the following codes in your python interactive shell

>> import tensorflow as tf

>> hello = tf.constant('Hello, TensorFlow!')

>> sess = tf.Session()

>> print(sess.run(hello))

If your system has successfully installed tensorflow, it will display "Hello, TensorFlow!".

Dataset:

Two datasets have been uploaded: the training image dataset and the entire image dataset (including both the training and test images).
You need to download those datasets in your local computer and when you run the programs, remember to change the path to your local 
directory.

In train426.py, you could also change the path to the entire dataset to see the accuracy and loss. 
Paths including "logs_train_dir" and logs_val_dir" are created to save tensorflow event files and visualize quantitive metrics.
