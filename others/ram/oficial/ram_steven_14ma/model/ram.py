import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import Model
from tensorflow.keras.layers import LSTMCell

from model.layers import ConvLayer
import time


class RecurrentAttentionModel(Model):
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
    def __init__(self, time_steps, n_glimpses, glimpse_size, num_classes, input_channels=1, max_gradient_norm = 5.0, std = 0.22):
        """
        Parameters
        ----------
        time_steps: int
            How many timestep the Core Network will have
        n_glimpses: int
            This decides the amount of glimpses the Glimpse Network will use
        glimpse_size : int
            This decides the size of the glimpses which the Glimpse Network uses
        
        num_classes: int
            How many classes in the dataset are
        
        input_channels: int (default 1)
            The amount of channels in the input image
        max_gradient_norm: float
            How much the gradient is clipped
        std: float
            Gaussian standard deviation for location network
        """
        super(RecurrentAttentionModel, self).__init__()
        self.n_glimpses = n_glimpses
        self.glimpse_size = glimpse_size
        self.time_steps = time_steps
        
        self.num_classes = num_classes

        self.max_gradient_norm = max_gradient_norm
        self.std = std
        # initialize glimpse network
        self.glimpse_network = GlimpseNetwork(n_glimpses=self.n_glimpses, glimpse_size=self.glimpse_size, input_channels=input_channels)
        # initialize our LSTM/RNN cell - hidden layers basically
        cell_size = 256
        self.cell = LSTMCell(cell_size) #LSTM(cell_size)
        # initialize the location network and baseline network
        self.location_network = LocationNetwork(cell_size=cell_size, std=self.std)
        self.baseline_netork = BaselineNetwork(cell_size=cell_size)
        # initialize the action network
        self.action_network = tf.keras.layers.Dense(self.num_classes)
        
        self.loc = None
        self.state = None
    
    def call(self, x):
        """
        This method is the output of the core network - a Recurrent Neural Network that maintains an 
        internal state that integrates information extracted from the history of past observations. 
        It encodes the agent's knowledge of the environment through multiple state vectors that get 
        updated at every time step t.
        
        Concretely, it takes the images as input, and combines it with its internal state h_{t-1} at the previous
        time step, to produce the new internal state `h_t` at the current time step.
        
        Parameters
        ----------
        x: tf.Variable
            This is a variable in which the images are fed into. It should have the shape of 
            [batch_size, image_height, image_width, image_channels]
            
        Returns
        -------
        rnn_outputs : tf.Tensor of type tf.float32
            A 2-D float tensor of shape [batch_size, 256] which is the output of the Core Network.
        """
        # we start with initializing the location, state, glimpses if they are None
        #if self.loc is None:
        self.loc = tf.random.uniform((x.shape[0], 2), minval=-1, maxval=1)
        self.state = self.cell.get_initial_state(batch_size=x.shape[0], dtype=tf.float32)
        glimpse = self.glimpse_network(x, self.loc)

        # calculating the RNN outputs
        self.locs, self.loc_means, rnn_outputs = [], [], []
        for i in range(self.time_steps):
            # calculate output and state
            output, self.state = self.cell(inputs=glimpse, states=self.state)
            #print(self.cell.variables)
            rnn_outputs.append(output)

            # calculating the next locations
            self.loc, loc_mean = self.location_network(output)
            self.locs.append(self.loc)
            self.loc_means.append(loc_mean)

            # get next glimpse
            glimpse = self.glimpse_network(x, self.loc)

        # convert list to one big tensor
        self.locs = tf.concat(axis=0, values=self.locs)
        self.locs = tf.reshape(self.locs, (self.time_steps, x.shape[0], 2))
        self.locs = tf.transpose(self.locs, [1, 0, 2])
        self.loc_means = tf.concat(axis=0, values=self.loc_means)
        self.loc_means = tf.reshape(self.loc_means, (self.time_steps, x.shape[0], 2))
        self.loc_means = tf.transpose(self.loc_means, [1, 0, 2])

        # calculate the baseline output for each rnn out
        self.baselines = []
        for output in rnn_outputs:
            self.baselines.append(self.baseline_netork(output))
        self.baselines = tf.stack(self.baselines)        # [timesteps, batch_sz]
        self.baselines = tf.transpose(self.baselines)   # [batch_sz, timesteps]
        return self.action_network(rnn_outputs[-1])
    
    @tf.function
    def predict(self, logits, labels):
        """
        This method returns the accuracy and the predicted class of a given logits
        
        Parameters
        ----------
        logits: tf.Tensor of type tf.float32
            The output of the RAM, which you can get by calling the call function
        labels: tf.Variable
            This is the variable in which the labels are fed into. It should have the shape of 
            [batch_size, num_classes]
        
        Returns
        -------
        accuracy, predictions: float32, tf.Tensor of type int
            A tuple with the first item beeing the accuracy and the second item beeing the predicted classes
        """
        prediction = tf.nn.softmax(logits)
        accuracy = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32)
        accuracy = tf.reduce_mean(accuracy)
        return accuracy, tf.argmax(prediction, 1), self.locs
    
    @tf.function
    def hybrid_loss(self, logits, labels):
        """
        This method returns the hybrid loss function of the Recurrent Attention Model.
        Note: The Location Network is the only one which is trained with the REINFORCE algorithm. Everything else is 
        trained with the cross entropy loss
        
        Parameters
        ----------
        logits: tf.Tensor of type tf.float32
            The output of the RAM, which you can get by calling the call function
        labels: tf.Variable
            This is the variable in which the labels are fed into. It should have the shape of 
            [batch_size, num_classes]
        
        Returns
        -------
        loss, classification_loss, reward, baselines_mse: float32, float32, float32, float32
            loss - the hybrid loss value
            classification_loss - the loss of the action network/classification layer
            reward - the reward of the whole RAM
            baseline_mse - the mean squared error of the baseline
        """
        # classification loss - we only need to classify at the last step of our RNN
        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        
        # RL reward
        prediction = tf.nn.softmax(logits)
        reward = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32)
        
        # baseline mse 
        rewards = tf.expand_dims(reward, 1)             # [batch_sz, 1]
        rewards = tf.tile(rewards, (1, self.time_steps))   # [batch_sz, timesteps]
        baselines_mse = tf.reduce_mean(tf.square((rewards - self.baselines)))        
    
        # REINFORCE loss
        advantages = rewards - tf.stop_gradient(self.baselines, name="stop_gradient_in_advantages")
        logll = self.log_likelihood(self.loc_means, self.locs, self.std)
        logllratio = -tf.reduce_mean(logll * advantages)
        
        # hybrid loss
        loss = logllratio + classification_loss + baselines_mse
        return loss, classification_loss, tf.reduce_mean(reward), baselines_mse      
    
    @tf.function
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
        gaussian = tfp.distributions.Normal(loc=loc_means, scale=std)
        # [timesteps, batch_sz, loc_dim]
        logll = gaussian._log_prob(x=locs)
        return tf.reduce_sum(logll, 2) # [batch_sz, timesteps]


class GlimpseNetwork(tf.keras.layers.Layer):
    """
    The Glimpse Network f_g gets as input the image X and  the coordinates l_{t-1} of the centre. 
    It produces a vector representation which includes has information about the glimpse g_t (from 
    the Glimpse Sensor) and the location l_{t-1}. 
    """
    
    def __init__(self, n_glimpses, glimpse_size, input_channels):
        """
        Parameters
        ----------
        n_glimpses: int
            This decides the amount of crops which will be returned
        glimpse_size : int
            This decides the size of the crops which will be returned
        input_channels: int
            The amount of channels in the input image
        """
        super(GlimpseNetwork, self).__init__()
        self.n_glimpses = n_glimpses
        self.glimpse_size = glimpse_size
        
        self.conv1_g = ConvLayer(128, self.glimpse_size, self.n_glimpses * input_channels, "conv1_g")
        self.conv1_l = ConvLayer(128, 1, 2, "conv1_l")
        self.out = ConvLayer(256, 1, 128, "information_concatination")
        
    @tf.function
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
        loc_reshaped = tf.reshape(loc, [batch_size, 1, 1, 2])
        
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
        #image = tf.reshape(image, (n_images, image_h, image_w, image_c))

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
                                                   name='glimpse_sensors')
            cur_glimpse = tf.image.resize(cur_glimpse,
                                          size=[crop_size, crop_size],
                                          method=tf.image.ResizeMethod.BICUBIC,
                                          antialias=True,
                                          name='glimpse_resize')
            glimpses.append(cur_glimpse)
            
        glimpses = tf.concat(glimpses, axis=-1)
        
        print(glimpses)
                
        return glimpses
    
class LocationNetwork(tf.keras.layers.Layer):
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
        super(LocationNetwork, self).__init__()
        self.std = std
        self.conv_l = ConvLayer(2, 1, cell_size, "conv_location")

    @tf.function
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
        core_output = tf.stop_gradient(cell_output, name="stop_gradient_core_network_in_location_network")
        batch_size, cell_size = core_output.shape

        # compute the next location, then impose noise
        mean = tf.reshape(core_output, [batch_size, 1, 1, cell_size])
        mean = self.conv_l(mean)
        mean = tf.reshape(mean, [batch_size, -1])
        mean = tf.clip_by_value(mean, -1, 1)

        # limit locs to [-1, 1] and stop gradient 
        loc = tf.maximum(-1.0, tf.minimum(1.0, mean + tf.random.normal(shape=mean.get_shape(), mean=0, stddev=self.std)))
        loc = tf.stop_gradient(loc, name="stop_gradient_of_locations")
        return loc, mean
    
    
class BaselineNetwork(tf.keras.layers.Layer):
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
        super(BaselineNetwork, self).__init__()
        self.conv_b = ConvLayer(1, 1, cell_size, "conv_baseline")
    
    @tf.function
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
        core_output = tf.stop_gradient(cell_output, name="stop_gradient_in_baseline")
        batch_size, cell_size = core_output.shape
        b = tf.reshape(core_output, [batch_size, 1, 1, cell_size])
        b = self.conv_b(b)
        b = tf.reshape(b, [batch_size, -1])
        b = tf.squeeze(b)
        return b
  
