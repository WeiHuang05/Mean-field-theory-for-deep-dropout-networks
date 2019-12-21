from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import io
from io import StringIO
from time import time

_Depth = 200
_Loop = 200
_Width = 1000
def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=(2.0/_Width)**0.5)
    return tf.Variable(initial)
   # initializer = tf.orthogonal_initializer()
   # return tf.Variable(initializer(shape)*0.5**0.5)
def bias_variable(shape):
    initial = tf.random_normal(shape, stddev = 0.05**0.5)
    return tf.Variable(initial)

input_images1= np.random.randn(1, _Width)*1.0
input_images2= np.random.randn(1, _Width)*1.0
correct_predic1= np.zeros((1,_Width))
correct_predic2= np.zeros((1,_Width))  
 
filename1 = 'ta_r9aa_g'
filename2 = 'ta_r9ab_g'
filename3 = 'ta_r9ab_s'
filename4 = 'tanh_r9aa_st'
filename5 = 'tanh_r9ab_st'
filename6 = 'tanh_r9abs_st'

file1_= open("{}.csv".format(filename1),'w')
file2_= open("{}.csv".format(filename2),'w')
file3_= open("{}.csv".format(filename3),'w')
file4_= open("{}.csv".format(filename4),'w')
file5_= open("{}.csv".format(filename5),'w')
file6_= open("{}.csv".format(filename6),'w')

y = tf.placeholder(tf.float32, [None, _Width])
x = tf.placeholder(tf.float32, [None, _Width])
kp_in = tf.placeholder(tf.float32)
kp  = tf.placeholder(tf.float32)

x_drop = tf.nn.dropout(x, keep_prob=kp_in)
h_fc_former_drop = x_drop
for i in range(_Depth):
    W_fc = weight_variable([_Width, _Width])
    b_fc = bias_variable([_Width])
    ha = tf.add(tf.matmul(h_fc_former_drop, W_fc), b_fc, name='pre%d' % i)
    h_fc = tf.nn.tanh(ha, name = 'pos%d' %i)
    #h_fc = ha
    h_fc_now_drop = tf.nn.dropout(h_fc, kp)
    h_fc_former_drop = h_fc_now_drop

outp = h_fc
# define the loss function
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(outp), reduction_indices=[1]))
mean_squared_error = tf.reduce_mean(tf.reduce_sum((outp-y)**2, reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
grads_and_vars = optimizer.compute_gradients(mean_squared_error)
grads,_ = list(zip(*grads_and_vars))

graph = tf.get_default_graph()
norms = tf.global_norm(grads)
init = tf.initialize_all_variables()
#print (tf.trainable_variables())
#var = [v for v in tf.trainable_variables() if v.name == "Variable_18:0"][0]
q_aa = np.zeros((_Loop,_Depth))
q_ab = np.zeros((_Loop,_Depth))
g_aa = np.zeros((_Loop,_Depth))
g_ab = np.zeros((_Loop,_Depth))
g_abs = np.zeros((_Loop,_Depth))
z1 = np.zeros((_Depth,_Width))
z2 = np.zeros((_Depth,_Width))

for i in range(_Loop):
    sess = tf.Session()
    print (i)
    sess.run(init)
    gda = sess.run(grads, feed_dict={x: input_images1, y: correct_predic1, kp_in: 1.0, kp: 0.9})
    gdb = sess.run(grads, feed_dict={x: input_images2, y: correct_predic2, kp_in: 1.0, kp: 0.9})
    
    for j in range(_Depth):
        g_abs[i,j] = np.mean(np.abs(gda[2*j]*gdb[2*j]))
        g_aa[i,j] = np.mean(gda[2*j]*gda[2*j])
        g_ab[i,j] = np.mean(gda[2*j]*gdb[2*j])

        if j%10==0:
            file4_.write(str(g_aa[i,j])+'\n') 
            file5_.write(str(g_ab[i,j])+'\n') 
            file6_.write(str(g_abs[i,j])+'\n')  
    sess.close()
  
for i in range(_Depth):
    file1_.write(str(np.mean(g_aa,0)[i])+'\n') 
    file2_.write(str(np.mean(g_ab,0)[i])+'\n') 
    file3_.write(str(np.mean(g_abs,0)[i])+'\n') 
       
