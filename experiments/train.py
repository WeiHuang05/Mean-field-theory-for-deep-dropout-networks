import numpy as np
import tensorflow as tf
from time import time
import math
from include.model import model
from tensorflow.examples.tutorials.mnist import input_data
import sys
# DATASET
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# PARAMS
_BATCH_SIZE = 1000
_STEP = 1000
_SAVE_PATH = "./tensorboard/cifar-10-v1/"
_Length = sys.argv

print(_Length)
  
filename = 'tanh2_orth_L10'
file_= open("{}.csv".format(filename),'w')   

x, y, sigma2, output, y_pred_cls, global_step = model()
print (sigma2)
print (x)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output),reduction_indices=[1])) 
# loss = tf.reduce_mean(tf.reduce_sum((y-output)**2,reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(loss, global_step=global_step)                                
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for sigw2_i in range(0,31):
    sigw2 = sigw2_i*0.1+1.0
    print (sigw2)
    global_accuracy = 0
    epoch_start = 0 
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init,feed_dict = {sigma2:sigw2}) 
    print ("The sigma2_w is %g " %(sigw2))
    for i in range(_STEP):
        input_images,corr_predic = mnist.train.next_batch(_BATCH_SIZE)
        i_global,_, batch_loss, batch_acc = sess.run([global_step,optimizer,loss,accuracy],
                        feed_dict={x:input_images, y:corr_predic})
        print ("step %d, training accuracy %g, loss %g" %(i_global, batch_acc, batch_loss) )

    file_.write(str(sigw2) + ' ' + str(batch_acc) + '\n')   
      #  test_accuracy = sess.run(accuracy,feed_dict={Sigma2:sigw2,x:mnist.test.images,y:mnist.test.labels})
    sess.close()    


