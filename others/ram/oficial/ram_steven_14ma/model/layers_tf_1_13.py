import tensorflow as tf

def get_weight(shape, name, trainable=True):
    #initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1)
    initial = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initial, trainable=trainable, name=name+'_W', dtype=tf.float32)

def get_bias(shape, name, trainable=True):
    """
    filter_height, filter_width, in_channels, out_channels]
    """
    return tf.Variable(tf.zeros(shape), trainable=trainable, name=name+'_b', dtype=tf.float32)

class ConvLayer():
    def __init__(self, n_filter, filter_size, input_channel, name):
        self.weights = get_weight([filter_size, filter_size, input_channel, n_filter], name)
        self.biases = get_bias([n_filter], name)
        self.name = name
    
    def __call__(self, x):
        with tf.variable_scope(self.name):
            conv = tf.nn.conv2d(x, self.weights, [1, 1, 1, 1], padding='VALID', data_format="NHWC")
            bias = tf.nn.bias_add(conv, self.biases)
            batch_norm = tf.layers.batch_normalization(bias, name=self.name + "_bn", reuse=tf.AUTO_REUSE)
            relu = tf.nn.relu(batch_norm, name=self.name+"_relu")
            return relu