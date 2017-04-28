# reference: tensorflow cifar10 tutorial
# build a simple ConvNet
import tensorflow as tf
import input426
'''522,775,4'''
    
def variable_on_cpu(name,shape,initilizer):
    with tf.device('/cpu:0'):
        dtype = tf.float32        
        var = tf.get_variable(name,shape, initializer = None,dtype = dtype) # get an existing variable or create a new one
    return var
    
def variable_of_weight(name,shape,stddev):
    dtype = tf.float32
    var = variable_on_cpu(name,shape,tf.truncated_normal_initializer
                                        (stddev=stddev,dtype = dtype))
    return var
    
#  Compute inferece on the inputs to make a prediction
def inference(images,batch_size,num_classes):

    #conv1, shape = [kernel size, kernel size,depth,kernel number]
    with tf.variable_scope('conv1') as scope:
        weights = variable_of_weight('weights', shape = [3,3,4,16],stddev = 5e-2)# 0.1
        biases = variable_on_cpu('biases',[16],tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images,weights,strides =[1,1,1,1],padding ='SAME')
        pre_activation = tf.nn.bias_add(conv, biases) # feature-add bias to conv
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
        
    #pool1 and norm1   
    # perform the max pooling on the conv1
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,4,1],strides=[1,2,2,1],
                            padding ='SAME',name = 'pool1')
    # local response normalization
    norm1 = tf.nn.lrn(pool1,4, bias=1.0, alpha =0.001/9.0,
                      beta = 0.75, name ='norm1')
                      
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = variable_of_weight('weights',shape = [3,3,16,16],stddev = 5e-2) #0.1
        biases = variable_on_cpu('biases',[16],tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1,weights,[1,1,1,1],padding = 'SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name =scope.name)
        
    #pool2 and norm2
    norm2 = tf.nn.lrn(conv2,4,bias = 1.0,alpha=0.001/9.0,beta =0.75,name='norm2')
    pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides =[1,1,1,1],
                            padding='SAME', name = 'pool2')

    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = variable_of_weight('weight', shape =[dim,128],stddev=0.005)
        biases = variable_on_cpu('biases',[128],tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+ biases,name = scope.name)

    #local4
    with tf.variable_scope('local4') as scope:
        weights = variable_of_weight('weights', shape=[128,128],stddev=0.005)
        biases = variable_on_cpu('biases',[128],tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name= scope.name)

    #softmax linear classifier
    with tf.variable_scope('softmax_linear') as scope:
        weights = variable_of_weight('softmax_linear', [128,num_classes], stddev=0.005)
        biases = variable_on_cpu('biases',[num_classes],tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases, name = scope.name)
    return softmax_linear

#  Compute the total loss of the prediction on the labels
def loss(logits, labels):
    with tf.variable_scope('loss')as scope:
        # compute sparse softmax cross entropy between logits and labels
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = labels,logits = logits, name = 'cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name,loss) # name + tensor
        return loss

#  Create an optimizer and apply to training dataset
def train(loss,learning_rate):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate)
    global_step = tf.Variable(0,name = 'global_step',trainable = False)
    train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op
  
# Evaluate
def evaluate(logits, labels):
    with tf.variable_scope('accuracy') as scope:        
        correct = tf.nn.in_top_k(logits,labels,1) # check whether in top 1 prediction
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name,accuracy)
    return accuracy
    
