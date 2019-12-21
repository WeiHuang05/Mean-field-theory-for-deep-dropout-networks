import tensorflow as tf
import sys

def weight_variable(shape,sig2):
   # initial = tf.truncated_normal(shape, stddev=0.01)
   # initial = tf.random_normal(shape, stddev=(2.0/400.0)**0.5)
   # return tf.Variable(initial)
   initializer = tf.orthogonal_initializer()
   return tf.Variable(initializer(shape)*sig2**0.5)

def bias_variable(shape):
   # initial = tf.constant(0.0, shape=shape)
    initial = tf.random_normal(shape, stddev = 0.05**0.5)
    return tf.Variable(initial)

def model( ):
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    _Width = 400
    _Layer = 10
    
    with tf.name_scope('main_params'):
        sigma2 = tf.placeholder(tf.float32,shape=[], name='sigma2')
        x = tf.placeholder(tf.float32, shape=[None,784], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        #x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
       # learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    #x = tf.reshape(x_image, [x_image.get_shape().as_list()[0], -1])
    x_drop = tf.nn.dropout(x, keep_prob=1)
    W_fc = weight_variable([784, _Width],sigma2)
    b_fc = bias_variable([_Width])
    h_fc = tf.nn.tanh(tf.matmul(x_drop, W_fc) + b_fc)
    h_fc1_drop = tf.nn.dropout(h_fc, keep_prob=1)

    num_layer = _Layer - 2
    h_fc_former_drop = h_fc1_drop

    for i in range(num_layer):
      W_fc = weight_variable([_Width, _Width],sigma2)
      b_fc = bias_variable([_Width])
      h_fc = tf.nn.tanh(tf.matmul(h_fc_former_drop, W_fc) + b_fc)
      h_fc_now_drop = tf.nn.dropout(h_fc, keep_prob=1)
      h_fc_former_drop = h_fc_now_drop

    W_fc = weight_variable([_Width, 10],sigma2)
    b_fc = bias_variable([10])
    softmax = tf.nn.softmax(tf.nn.tanh(tf.matmul(h_fc_former_drop, W_fc)) + b_fc)      
    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, sigma2, softmax, y_pred_cls, global_step


#def lr(lri):
#    learning_rate = 10**(-1-1.0/100*lri)
#    return learning_rate
