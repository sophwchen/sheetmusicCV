import numpy as np
import matplotlib.pyplot as plt
import mygrad as mg
#%matplotlib notebook
from mynn.layers.dense import dense
from mygrad.nnet.initializers.he_normal import he_normal
from mygrad.nnet.activations.relu import relu
from mynn.optimizers.sgd import SGD
from mygrad.nnet.losses import softmax_crossentropy

from mynn.layers.conv import conv
from mynn.layers.dense import dense

from mygrad.nnet.initializers import glorot_uniform
from mygrad.nnet.activations import relu
from mygrad.nnet.layers import max_pool
from mygrad.nnet.losses import softmax_crossentropy

class Model:
    ''' A simple convolutional neural network. '''
    def __init__(self, num_input_channels, f1, f2, d1, num_classes, drop=0.5):
        """
        Parameters
        ----------
        num_input_channels : int
            The number of channels for a input datum
            
        f1 : int
            The number of filters in conv-layer 1
        
        f2 : int
            The number of filters in conv-layer 2

        d1 : int
            The number of neurons in dense-layer 1
        
        num_classes : int
            The number of classes predicted by the model.
        """
        self.drop = 1-drop

        init_kwargs = {'gain': np.sqrt(2)}
        self.conv1 = conv(num_input_channels, f1, 5, 5, weight_initializer=glorot_uniform, weight_kwargs=init_kwargs)
        self.conv2 = conv(f1, f2, 5, 5 ,weight_initializer=glorot_uniform, weight_kwargs=init_kwargs)
        #self.dense1 = dense(f2 * 144 * 148, d1, weight_initializer=glorot_uniform, weight_kwargs=init_kwargs)
        self.dense1 = dense(f2 * 484, d1, weight_initializer=glorot_uniform, weight_kwargs=init_kwargs)
        self.dense2 = dense(d1, num_classes, weight_initializer=glorot_uniform, weight_kwargs=init_kwargs)


    def __call__(self, x, test=False):
        ''' Defines a forward pass of the model.
        
        Parameters
        ----------
        x : numpy.ndarray, shape=(N, 1, 32, 32)
            The input data, where N is the number of images.
            
        Returns
        -------
        mygrad.Tensor, shape=(N, num_classes)
            The class scores for each of the N images.
        '''
        
        # Define the "forward pass" for this model based on the architecture detailed above.
        # Note that, to compute 
        # We know the new dimension given the formula: out_size = ((in_size - filter_size)/stride) + 1
        #print(x.shape)
        # <COGINST>
        x = relu(self.conv1(x))
        
        if not test:
            # only want to apply dropout during the training phase
            dropout1 = np.random.binomial(1, self.drop, size=x.shape) / self.drop
            x = x * dropout1
        
        #print(x.shape)
        x = max_pool(x, (2, 2), 1)
        #print(x.shape)
        x = relu(self.conv2(x))
        
        if not test:
            dropout2 = np.random.binomial(1, self.drop, size=x.shape) / self.drop
            x = x * dropout2
        
       # print(x.shape)
        x = max_pool(x, (2, 2), 1)
        x = relu(self.dense1(x.reshape(x.shape[0], -1)))
        
        if not test:
            dropout3 = np.random.binomial(1, self.drop, size=x.shape) / self.drop
            x = x * dropout3
        
        return self.dense2(x)
        # </COGINST>

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model. """
        # Create a list of every parameter contained in the 4 layers you wrote in your __init__ function
        # <COGINST>
        params = []
        for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
            params += list(layer.parameters)
        return params
        # </COGINST>