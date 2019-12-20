import theano.tensor as T
import theano
#import os; os.environ['KERAS_BACKEND'] = 'theano'
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
from scipy.stats import ortho_group
import scipy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.noise import GaussianDropout
from keras.initializers import normal, orthogonal
import keras.backend as K
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from keras.layers.advanced_activations import LeakyReLU, PReLU
import tensorflow as tf

def hard_tanh(xx):
    return K.minimum(K.relu(xx + 1) - 1, 1)

def erf2(x):
    return tf.erf(np.sqrt(np.pi)/2*x)    
     
def randomize_weight(w, mu=None, sigma=None):
    if mu is None:
        mu = w.mean()
    if sigma is None:
        sigma = w.std()
    return sigma * np.random.randn(*w.shape) + mu

def randomize_weights(weights, bias_sigma=0.1, weight_sigma=1.0):
    random_weights = []
   # print (bias_sigma)
    for w in weights:
        if w.ndim == 1:
            rw = randomize_weight(w, mu=0, sigma=bias_sigma)
            rw = -rw
        else:
            sigma = weight_sigma * 1.0 / np.sqrt(w.shape[0])
            rw = randomize_weight(w, mu=0, sigma=sigma)
        random_weights.append(rw)
     
    return random_weights

class RandNet(object):
    """Simple wrapper around Keras model that throws in some useful functions like randomization"""
    def __init__(self, input_dim, n_hidden_units, n_hidden_layers,  rho = 0.6, nonlinearity='tanh', 
            init = 'gau', bias_sigma=0.0, weight_sigma=1.25, input_layer=None, flip=False, output_dim=None):
        #if input_layer is not None:
        #    assert input_layer.output_shape[1] == input_dim
        self.input_dim = input_dim
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.nonlinearity = nonlinearity
        self.bias_sigma = bias_sigma
        self.weight_sigma = weight_sigma
        self.input_layer = input_layer

        if output_dim is None:
            output_dim = n_hidden_units
        self.output_dim = output_dim

        model = Sequential()
        get_custom_objects().update({'hard_tanh': Activation(hard_tanh)})
        get_custom_objects().update({'erf2': Activation(erf2)})
        if input_layer is not None:
            model.add(input_layer)
           # model.add(Dropout(0.1, input_layer))
         
        #model.add(Activation('tanh'))    
        for i in range(n_hidden_layers):
            nunits = n_hidden_units if i < n_hidden_layers - 1 else output_dim
            if flip:
                if nonlinearity=='prelu':
                    model.add(LeakyReLU(alpha = 0.5, input_shape=(input_dim,), name='a%d'%i))
                else:
                    model.add(Activation(nonlinearity, input_shape=(input_dim,), name='a%d'%i))

               # model.add(Activation(nonlinearity, input_dim=1000, name='a%d'%i))
                model.add(Dropout(1-rho))  # dropout = 1 - rho
                if init =='gau':
                    model.add(Dense(nunits, name='d%d'%i,
                               kernel_initializer=normal(mean=0.0,stddev=weight_sigma*1.0/np.sqrt(n_hidden_units)), 
                               bias_initializer=normal(mean=0.0,stddev=bias_sigma)))
                if init =='orth':
                    model.add(Dense(nunits, name='d%d'%i,
                               kernel_initializer=orthogonal(gain=weight_sigma), 
                               bias_initializer=normal(mean=0.0,stddev=bias_sigma)))               
               # model.add(Dense(nunits, name='d%d'%i))
            else:
                model.add(Dense(nunits, input_shape=(input_dim,), name='%d'%i))
                if i <  n_hidden_layers - 1 or self.output_dim == self.n_hidden_units:
                    model.add(Activation(nonlinearity, name='a%d'%i))
                else:
                    # Theano is optimizing out the nonlinearity if it can which is breaking shit
                    # Give it something that it won't optimize out.
                    model.add(Activation(lambda x: T.minimum(x, 999999.999),  name='a%d'%i))

        model.build()
        self.model = model
        # print(self.hs)
        
        self.weights = model.get_weights()
        self.dense_layers = filter(lambda x:  x.name.startswith('d'), model.layers)
        self.activ_layers = filter(lambda x:  x.name.startswith('a'), model.layers)

        self.hs = [h.output for h in self.dense_layers]
        self.ac = [b.output for b in self.activ_layers]

        self.f_acts = self.f_jac = self.f_jac_hess = self.f_act = None
        vec = K.ones_like(self.model.input)

    def compile(self, jacobian=False):
        #self.model.compile('adagrad', 'mse')
        self.f_acts = K.function([self.model.input], self.hs)

    def get_acts(self, xs):
        if self.f_acts is None:
            self.f_acts = K.function([self.model.input, K.learning_phase()], self.hs)
           # self.f_acts = K.function([self.model.input], self.hs)
        #return self.f_acts((xs,))
        return self.f_acts([xs,1])

    def get_act(self, xs):
        if self.f_act is None:
            self.f_act = K.function([self.model.input, K.learning_phase()], self.hs[-1])
            #self.f_act = K.function([self.model.input], self.hs[-1])
        #return self.f_act((xs,))
        return self.f_act([xs,1])

    def Gaussian_randomize(self, bias_sigma=0.0, weight_sigma=1.0):
        """Randomize the weights and biases in a model.

        Note this overwrites the current weights in the model.
        """
        self.model.set_weights(randomize_weights(self.weights, bias_sigma, weight_sigma))

    def Orthogonal_randomize(self, bias_sigma=None, weight_sigma=None):
        """Randomize the weights and biases in a model.

        Note this overwrites the current weights in the model.
        """
        if bias_sigma is None:
            bias_sigma = self.bias_sigma
        if weight_sigma is None:
            weight_sigma = self.weight_sigma
        w0 = randomize_weights(self.weights, bias_sigma, weight_sigma)  
        for lid in range(0, self.n_hidden_layers): 
            w0[2*lid] = ortho_group.rvs(dim=self.n_hidden_units)*weight_sigma
            #print (w0[2*lid].shape)
        self.model.set_weights(w0)

# construct erf networks by myself
class DiyNet(object):
    """Simple wrapper around Keras model that throws in some useful functions like randomization"""
    def __init__(self, input_dim, n_hidden_units, n_hidden_layers, rho = 0.6, nonlinearity='erf', 
           bias_sigma=0.0, weight_sigma=1.25, input_layer=None, flip=False, output_dim=None):
        
        self.input_dim = input_dim
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.nonlinearity = nonlinearity
        self.bias_sigma = bias_sigma
        self.weight_sigma = weight_sigma
        self.input_layer = input_layer
        self.rho = rho

        if output_dim is None:
            output_dim = n_hidden_units
        self.output_dim = output_dim

    def get_act(self, xs, nonlinearity='erf'):

        weight, bias = randomize_weights_diy(self)
        
       # print (np.var(bias))
        hs = np.zeros((xs.shape[0],self.n_hidden_units))
        for i in range(xs.shape[0]):
            Be = np.random.binomial(1, self.rho, self.n_hidden_units)
            Be_matrix = np.diag(Be)
            post_active = scipy.special.erf(np.sqrt(np.pi)/2*xs[i,:])
            weight1 = np.dot(weight, Be_matrix)
            pre_active = np.dot(post_active,weight1)/self.rho + bias
            hs[i,:] = pre_active
        return hs
       # print (pre_active.shape)    
        
def randomize_weights_diy(self):
        
    width = self.n_hidden_units
    weights = self.weight_sigma *1.0 / np.sqrt(width) * np.random.randn(width,width)
    biases = self.bias_sigma * np.random.randn(width)

    return weights, biases    


    

   

