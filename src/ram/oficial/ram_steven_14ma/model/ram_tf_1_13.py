import tensorflow as tf
from tensorflow_addons.seq2seq import decoder as rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow.compat.v1.nn.rnn_cell import LSTMCell

from model.layers_tf_1_13 import ConvLayer
import time


class RecurrentAttentionModel():
    """
    A Recurrent Attention Model ist a recurrent neural network that processes inputs sequentially, 
    attending to different locations within the image one at a time, and incrementally
    combining information from these fixations to build up a dynamic internal representation of the image.
    
    Is it inspired by the human perception because that one does not tend to process a whole scene
    in its entirety at once. Instead humans focus attention selectively on parts of the visual space to
    acquire information when and where it is needed, and combine information from different fixations
    over time to build up an internal representation of the scene, guiding future eye movements
    and decision making.
    """
    def __init__(self, time_steps, n_glimpses, glimpse_size, learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch, max_gradient_norm = 5.0, std = 0.22):
        """
        Parameters
        ----------
        time_steps: int
            How many timestep the Core Network will have
        n_glimpses: int
            This decides the amount of glimpses the Glimpse Network will use
        glimpse_size : int
            This decides the size of the glimpses which the Glimpse Network uses
            
        learning_rate: tf.placeholder, tf.Constant or tf.Variable or float
            How big the gradient will be subtracted from the weights alias the learning rate
        learning_rate_decay_factor: float
            How much the learning rate decays exponentially
        min_learning_rate: float
            How big the smallest learning rate will be
        training_steps_per_epoch: int
            How many training steps will be in each epoch
            
        max_gradient_norm: float
            How much the gradient is clipped
        std: float
            Gaussian standard deviation for location network
        """
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.training_steps_per_epoch = training_steps_per_epoch
        self.min_learning_rate = min_learning_rate

        self.n_glimpses = n_glimpses
        self.glimpse_size = glimpse_size
        self.time_steps = time_steps

        self.max_gradient_norm = max_gradient_norm
        self.std = std
        
    def build(self, X_PH, Y_PH):
        """
        This methods creates the Recurrent Attention Model. 
        
        You can train this model by using self.train_op.
        
        The loss functions are:
            self.loss - the hybrid loss
            self.classification_loss - the loss of the action network
            self.baselines_mse - the loss of the baseline network
        
        Parameters
        ----------
        X_PH: tf.Placeholder
            This is the placeholder in which the images are fed into. It should have the shape of 
            [batch_size, image_height, image_width, image_channels]
        Y_PH: tf.Placeholder
            This is the placeholder in which the labels are fed into. It should have the shape of 
            [batch_size, num_classes]
        """
        start = time.time()
        core_net_output = self.get_core_network(X_PH)
        
        # classification loss - we only need to classify at the last step of our RNN
        last_output = core_net_output[-1]
        logits = tf.layers.dense(last_output, Y_PH.shape[-1].value, name="action_network_out")
        self.classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_PH, logits=logits))
        
        # RL reward
        self.prediction = tf.nn.softmax(logits)
        reward = tf.cast(tf.equal(tf.argmax(self.prediction, 1), tf.argmax(Y_PH, 1)), tf.float32)
        self.reward = tf.reduce_mean(reward)
    
        # baseline mse 
        rewards = tf.expand_dims(reward, 1)             # [batch_sz, 1]
        rewards = tf.tile(rewards, (1, self.time_steps))   # [batch_sz, timesteps]
        self.baselines_mse = tf.reduce_mean(tf.square((rewards - self.baselines)))        
    
        # REINFORCE loss
        advantages = rewards - tf.stop_gradient(self.baselines)
        logll = self.log_likelihood(self.loc_means, self.locs, self.std)
        logllratio = -tf.reduce_mean(logll * advantages)
        
        # hybrid loss
        self.loss = logllratio + self.classification_loss + self.baselines_mse
        
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.learning_rate, global_step, self.training_steps_per_epoch, self.learning_rate_decay_factor, staircase=True)
        self.lr = tf.maximum(lr, self.min_learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=global_step)
        print("model build, took:", time.time() - start)
            
    def get_core_network(self, X_PH):
        """
        An Recurrent Neural Network that maintains an internal state that integrates information extracted from the
        history of past observations. It encodes the agent's knowledge of the environment through one or multiple
        state vectors that get updated at every time step t.
        
        Concretely, it takes the images as input, and combines it with its internal state h_{t-1} at the previous
        time step, to produce the new internal state `h_t` at the current time step.
        
        Parameters
        ----------
        X_PH: tf.Placeholder
            This is the placeholder in which the images are fed into. It should have the shape of 
            [batch_size, image_height, image_width, image_channels]
            
        Returns
        -------
        rnn_outputs : tf.Tensor of type tf.float32
            A 2-D float tensor of shape [batch_size, 256] which is the output of the Core Network.
        """
        # initialize glimpse network
        self.glimpse_network = GlimpseNetwork(n_glimpses=self.n_glimpses, glimpse_size=self.glimpse_size)
        
        # initialize our LSTM/RNN cell - hidden layers basically
        cell_size = 256
        cell = LSTMCell(cell_size) #LSTM(cell_size)
        
        # initialize the location network and baseline network
        self.location_network = LocationNetwork(cell_size=cell_size, std=self.std)
        self.baseline_netork = BaselineNetwork(cell_size=cell_size)
        
        # core network - we build the network here
        # we start with initializing the location, state, glimpses
        init_loc = tf.random_uniform((X_PH.shape[0].value, 2), minval=-1, maxval=1)
        init_state = cell.zero_state(X_PH.shape[0].value, tf.float32)
        init_glimpse = self.glimpse_network(X_PH, init_loc)
               
        # calculate the rnn outputs
        self.locs, self.loc_means, self.baselines = [], [], []
        def loop_function(prev, _):
            loc, loc_mean = self.location_network(prev)
            self.locs.append(loc)
            self.loc_means.append(loc_mean)
            glimpse = self.glimpse_network(X_PH, loc)
            return glimpse
        rnn_inputs = [init_glimpse]
        rnn_inputs.extend([0] * self.time_steps)
        rnn_outputs, _ = rnn_decoder(rnn_inputs, init_state, cell, loop_function=loop_function)
        
        # calculate the baseline output for each rnn out
        for output in rnn_outputs[1:]:
            self.baselines.append(self.baseline_netork(output))
        self.baselines = tf.stack(self.baselines)        # [timesteps, batch_sz]
        self.baselines = tf.transpose(self.baselines)   # [batch_sz, timesteps]
        
        # convert list to one big tensor
        self.locs = tf.concat(axis=0, values=self.locs)
        self.locs = tf.reshape(self.locs, (self.time_steps, X_PH.shape[0].value, 2))
        self.locs = tf.transpose(self.locs, [1, 0, 2])
        self.loc_means = tf.concat(axis=0, values=self.loc_means)
        self.loc_means = tf.reshape(self.loc_means, (self.time_steps, X_PH.shape[0].value, 2))
        self.loc_means = tf.transpose(self.loc_means, [1, 0, 2])
        return rnn_outputs         
    
    def log_likelihood(self, loc_means, locs, std):
        """
        This method creates a normal distribution with a given mean (mu=loc_means) and 
        a standard deviation (std=std). It will use the created distribution in order to 
        return the log likelihood of the given locations (locs)
        
        Parameters
        ----------
        loc_means: List of len=num_glimpses with Tensors of Shape [batch_size, 2]
             a list with  contains tensors with shape (B, 2)
        locs: List of len=num_glimpses with Tensors of Shape [batch_size, 2]
             a list with all sampled location of all timesteps
        std: float
            Gaussian standard deviation for location network
        
        Returns
        -------
        logll : tf.Tensor of type tf.float32
            A 2-D float tensor of shape [batch_size, timesteps]
        """
        # [timesteps, batch_sz, loc_dim]
        loc_means = tf.stack(loc_means)
        locs = tf.stack(locs)
        gaussian = Normal(loc_means, std)
        # [timesteps, batch_sz, loc_dim]
        logll = gaussian._log_prob(x=locs)
        return tf.reduce_sum(logll, 2) # [batch_sz, timesteps]


class GlimpseNetwork():
    """
    The Glimpse Network f_g gets as input the image X and  the coordinates l_{t-1} of the centre. 
    It produces a vector representation which includes has information about the glimpse g_t (from 
    the Glimpse Sensor) and the location l_{t-1}. 
    """
    
    def __init__(self, n_glimpses, glimpse_size):
        """
        Parameters
        ----------
        n_glimpses: int
            This decides the amount of crops which will be returned
        glimpse_size : int
            This decides the size of the crops which will be returned
        """
        self.n_glimpses = n_glimpses
        self.glimpse_size = glimpse_size
        
        self.conv1_g = ConvLayer(128, self.glimpse_size, self.n_glimpses, "conv1_g")
        self.conv1_l = ConvLayer(128, 1, 2, "conv1_l")
        self.out = ConvLayer(256, 1, 128, "information_concatination")
        
    def __call__(self, image, loc):
        """
        Gets a image and locations and produces a vectorial reprensentation of rhem.
        
        Parameters
        ----------
        image : tf.Tensor of type tf.float32 or tf.unit8
            Tensorflow Tensor in which represents the images (only a single image a time).
            It should have 4 Dimensions with: [batch_size, image_height, image_width, n_channels]
            e.g. if you are using the MNIST dataset the placeholder should have the shape [batch_size, 28, 28, 1]
      
        loc : tf.Tensor of type tf.float32
            A 2-D float tensor of shape [1, 2]. The coordinates range from -1.0 to 1.0. 
            The coordinates (-1.0, -1.0) correspond to the upper left corner, 
            the lower right corner is located at (1.0, 1.0) and the center is at (0, 0).

        Returns
        -------
        tf.Tensor of the same type as image
            A 2-D Tensor of Shape [batch_size, 256] with type tf.float32
        """
        glimpses = self.get_glimpses(image, loc, self.n_glimpses, self.glimpse_size)
        
        batch_size, _, _, _ = image.shape
        loc_reshaped = tf.reshape(loc, [batch_size.value, 1, 1, 2])
        
        x_1 = self.conv1_g(glimpses)
        x_2 = self.conv1_l(loc_reshaped)
        x = self.out(x_1 + x_2)
        x = tf.reshape(x, [batch_size, -1])
        return x
        
    def get_glimpses(self, image, loc, n_crops, crop_size):
        """
        The GlimpseSensor extracts a crop p(x_t, l_{t-1}) out of a input image centered 
        at the coordinates l_{t-1}. The parameter crop_h and crop_w decide the size (width and height) 
        of the crops and n_crops decides how many crops will be extracted.
        
        Parameters
        ----------
        image : tf.Tensor of type tf.float32 or tf.unit8
            Tensorflow Tensor in which represents the images/batches of images.
            It should have 4 Dimensions with: [batch_size, image_height, image_width, n_channels]
            e.g. if you are using the MNIST dataset the placeholder should have the shape [batch_size, 28, 28, 1]
        loc : tf.Tensor of type tf.float32
            A 2-D integer tensor of shape [batch_size, 2] with coordinates with range from -1.0 to 1.0. 
            The coordinates (-1.0, -1.0) correspond to the upper left corner, 
            the lower right corner is located at (1.0, 1.0) and the center is at (0, 0).
        n_crops: int
            This decides the amount of crops which will be returned
        crop_size : int
            This decides the size of the crops which will be returned
        
        Returns
        -------
        glimpses: tf.Tensor same type as image
            The extracted climpses with the shape of [n_crops, crop_size, crop_size, channel_size]
        """
        n_images, image_h, image_w, image_c = image.shape
        image = tf.reshape(image, (n_images, image_h, image_w, image_c))

        # pad the input image
        max_size = crop_size * (2 ** (n_crops - 1))
        image = tf.pad(image, [[0, 0], [max_size, max_size], [max_size, max_size], [0, 0]], 'CONSTANT') 

        glimpses = []
        for i in range(n_crops):
            current_size = int(crop_size * (2 ** (i)))
            # extract glimpses
            cur_glimpse = tf.image.extract_glimpse(image,
                                                   size=[current_size, current_size],
                                                   offsets=loc,
                                                   centered=True,
                                                   normalized=True,
                                                   uniform_noise=True,
                                                   name='glimpse_sensor')
            cur_glimpse = tf.image.resize_images(cur_glimpse,
                                                 size=[crop_size, crop_size],
                                                 method=tf.image.ResizeMethod.BILINEAR,
                                                 align_corners=False)
            glimpses.append(cur_glimpse)
        glimpses = tf.concat(glimpses, axis=-1)
        return glimpses
    
class LocationNetwork():
    """
    The Location Networks selects a the new location for the Glimpse Network. 
    It uses the internal state $h_t$ of the core network to produce the location coordinates l_t
    for the next time step. 

    NOTE FOR FUTURE: It is trained with the REINFORCE algorithm while on the other hand 
    everything else is trained with a normal cross entropy loss.
    """

    def __init__(self, cell_size, std=0.22):
        """
        Parameters
        ----------
        cell_size : int
            The output dimension of the core network
        std : float
            Gaussian standard deviation for location network
        """
        self.std = std
        self.conv_l = ConvLayer(2, 1, cell_size, "conv_location")
        
    def __call__(self, cell_output):
        """
        Gets the output of the core network and produces new location coordinates
        
        Parameters
        ----------
        cell_output : tf.Tensor of type tf.float32
            A 2-D float from the core network with size [batch_size, cell_size]. 
            It represents the output from the RNN or LSTM cell
            
        Returns
        -------
        loc : tf.Tensor of type tf.float32
            A 2-D float tensor of shape [1, 2]. The coordinates range from -1.0 to 1.0. 
        """
        # stop gradient flow
        core_output = tf.stop_gradient(cell_output)
        batch_size, cell_size = core_output.shape

        # compute the next location, then impose noise
        mean = tf.reshape(core_output, [batch_size.value, 1, 1, cell_size.value])
        mean = self.conv_l(mean)
        mean = tf.reshape(mean, [batch_size, -1])
        mean = tf.clip_by_value(mean, -1, 1)
        
        # limit locs to [-1, 1] and stop gradient 
        loc = tf.maximum(-1.0, tf.minimum(1.0, mean + tf.random_normal(mean.get_shape(), 0, self.std)))
        loc = tf.stop_gradient(loc)
        return loc, mean
    
    
class BaselineNetwork():
    """
    The BaseLineNetwork is the baseline which is used in order to recude the variance of the gradient.
    This time of baseline is also known as the value function in the reinforce learning literatur.
    """
        
    def __init__(self, cell_size):
        """
        Parameters
        ----------
        cell_size : int
            The output dimension of the core network
        """
        self.conv_b = ConvLayer(1, 1, cell_size, "conv_baseline")

    def __call__(self, cell_output):
        """
        Gets the output of the core network and produces the output of the basline network
        
        Parameters
        ----------
        cell_output : tf.Tensor of type tf.float32
            A 2-D float from the core network with size [batch_size, cell_size]. 
            It represents the output from the RNN or LSTM cell
        
        Returns
        -------
        b : tf.Tensor of type tf.float32
            A 2-D float tensor of shape [batch_size, 2]. The coordinates range from -1.0 to 1.0. 
        """
        # the next location is computed by the location network
        core_output = tf.stop_gradient(cell_output)
        batch_size, cell_size = core_output.shape
        b = tf.reshape(core_output, [batch_size.value, 1, 1, cell_size.value])
        b = self.conv_b(b)
        b = tf.reshape(b, [batch_size, -1])
        b = tf.squeeze(b)
        return b
  

