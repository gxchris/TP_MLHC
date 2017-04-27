import tensorflow as tf
import numpy as np
import os
import input426
import model426

## run training set
# Constants
NUM_CLASSES = 2
IMGH = 522
IMGW = 775
BATCH_SIZE = 8
CAPACITY = 1000
MAX_STEP = 6000 # generally >10000
RATIO = 0.2
learning_rate = 0.0001

def run_training():
    # read image data
    path = "C:/Users/Xin/Desktop/S17/94867/Dataset/train"
    logs_train_dir = "C:/Users/Xin/Desktop/S17/94867/Dataset/log/train"
    logs_val_dir = "C:/Users/Xin/Desktop/S17/94867/Dataset/log/val"
    train_img,train_label,val_img,val_label = input426.import_files(path,RATIO)
    tra_img_batch,tra_label_batch = input426.get_batch(train_img,train_label,
                                    IMGH,IMGW,BATCH_SIZE,CAPACITY)
    val_img_batch,val_label_batch = input426.get_batch(val_img,val_label,
                                    IMGH,IMGW,BATCH_SIZE,CAPACITY)
    
    # compute the logits predictions from the reference model
    logits = model426.inference(tra_img_batch,BATCH_SIZE,NUM_CLASSES)
    loss = model426.loss(logits, tra_label_batch)
    train_op = model426.train(loss, learning_rate)
    acc= model426.evaluate(logits,tra_label_batch)
    
    x = tf.placeholder(tf.float32,shape = [BATCH_SIZE,IMGH,IMGW,4])
    y = tf.placeholder(tf.int16, shape = [BATCH_SIZE])
    
    with tf.Session() as sess:
        saver = tf.train.Saver() # create a saver   
        sess.run(tf.global_variables_initializer()) # return an ops that initialize global variables
        coord = tf.train.Coordinator() # create a coordinator
        # start all queue runner
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        # summary for tensorboard visualization
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
        val_writer = tf.summary.FileWriter (logs_val_dir,sess.graph)
        
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                tra_images, tra_labels = sess.run([tra_img_batch,tra_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                            feed_dict={x:tra_images,y:tra_labels})
                    
                if step % 5 == 0:
                    print('Step %d, train loss = %.2f, accuracy = %.2f%%' 
                            %(step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str,step)
                    
                if step %200 == 0 or (step+1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_img_batch,val_label_batch])
                    val_loss,val_acc = sess.run([loss,acc],
                                                feed_dict = {x:val_images,y:val_labels})
                    print('VAL-Step%d, val loss= %.2f accuracy = %.2f%%'  
                           %(step,val_loss,val_acc*100.0))         
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str,step)
                    
                if step % 2500 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
                    
        except tf.errors.OutOfRangeError:
            
            print('Training is done')
            
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

    
