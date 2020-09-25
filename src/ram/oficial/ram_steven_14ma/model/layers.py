import tensorflow as tf

def get_weight(shape, name, trainable=True):
    """
    This method initializes randomly the weights with a min and max value of 0.1
    
    Parameters
    ----------
    shape: List
        The shape of the weight
    name: String
        The name of the weight
    trainable: boolean (default True)
        If its False then the variables are not trainable
    
    Returns
    ----------
    weight: tf.Variable
        The weight with the given configuration    
    """
    initial = tf.random.uniform(shape, minval=-0.1, maxval = 0.1)
    #initial = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initial, trainable=trainable, name=name+'_W', dtype=tf.float32)

def get_bias(shape, name, trainable=True):
    """
    This method initializes the bias with a constant value of 0
    
    Parameters
    ----------
    shape: List
        The shape of the weight
    name: String
        The name of the weight
    trainable: boolean (default True)
        If its False then the variables are not trainable
    
    Returns
    ----------
    bias: tf.Variable
        The bias with the given configuration    
    """
    return tf.Variable(tf.zeros(shape), trainable=trainable, name=name+'_b', dtype=tf.float32)

class ConvLayer(tf.keras.layers.Layer):
    """
    A Convolution layer 
    """
    def __init__(self, n_filter, filter_size, input_channel, name, batch_norm=True):      
        """
        This method initializes the convolution layer
            
        Parameters
        ----------
        n_filter: int
            The amount of filter which the convolution will use (alias the output channel size)
        filter_size: int
            The size of a single kernel of the convolution layer
        input_channel: int
            The channel size of the input
        name: String
            The name of the convolution layer
        """  
        super(ConvLayer, self).__init__()
        self.weight = get_weight([filter_size, filter_size, input_channel, n_filter], name)
        self.bias = get_bias([n_filter], name)
        self.n = name
        
        if batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization(name=self.n + "_batch_norm")
        else:
            self.batch_norm = None
    
    def __call__(self, x):
        """
        Calculates the output of the convolution layer. 

        Parameters
        ----------
        x: np.array
            input images
        
        Returns
        ----------
        out: tf.Tensor
            the output tensor. The convolution layer is followed by a batch_norm layer and a relu layer.
        """
        conv = tf.nn.conv2d(x, self.weight, [1, 1, 1, 1], padding='VALID', data_format="NHWC", name=self.n + "_convolution")
        add_bias = tf.nn.bias_add(conv, self.bias, name=self.n + "_add_bias")
        if self.batch_norm:
            add_bias = self.batch_norm(add_bias)
        relu = tf.nn.relu(add_bias, name=self.n + "_relu")
        return relu
