"""
Configuration File to Classify the standard MNIST Dataset
using the Recurrent Attention Model, presented in

Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
"Recurrent models of visual attention."
Advances in neural information processing systems. 2014.

Author: Sascha Fleer
"""

from MNIST_experiment import Experiment

class MNIST_DOMAIN_OPTIONS:
    """
    Class for the Setup of the Domain Parameters
    """
    # Size of each image: MNIST_SIZE x MNIST_SIZE
    MNIST_SIZE = 28
    #
    #   ================
    #   Reward constants
    #   ================
    #   Reward for correctly Identifying a number:
    REWARD = +1.
    #   Step Reward

    #   ======================
    #   Domin specific options
    #   ======================
    #
    # Number of image channels: 1
    # --> greyscale
    CHANNELS = 1
    #
    # Resolution of the Sensor
    SENSOR = 8
    # Depth of glimpse (depth = 1--> original solution)
    DEPTH = 1
    # Scaling Factor of zooms
    SCALING_FACTOR = 2
    # Number of Glimpses
    NGLIMPSES = 7
    # Standard Deviation of the Location Policy
    LOC_STD = 0.11
    # Defines how far the center of the glimpse
    # can reach the borders
    # UNIT_PIXELS = 13 --> g_c \in [2,27] for 28x28x1 images
    UNIT_PIXELS = 13
    # Translated MNIST
    TRANSLATE = False
    # Size of each image: MNIST_SIZE x MNIST_SIZE
    TRANSLATED_MNIST_SIZE = 60

class PARAMETERS:
    """
    Class for specifying the parameters for
    the learning algorithm
    """

    #   =========================
    #   General parameters for the
    #   experiment
    #   =========================

    #   Number of learning epochs
    MAX_EPOCHS= 2000
    #   Batch size
    BATCH_SIZE = 20
    #   Early stopping
    EARLY_STOPPING = True
    #   Number of Epochs observing the worsening of
    #   Validation set, before stopping
    PATIENCE = 100

    #   =========================
    #   Save and Load the Model Weights
    #   =========================
    LOAD_MODEL = False
    MODEL_FILE_PATH = './MNIST_Results/'
    MODEL_FILE = 'network.h5'


    #   =========================
    #   Algorithm specific parameters
    #   =========================

    #   To be used optimizer:
    #   rmsprop
    #   adam
    #   adadelta
    #   sgd
    OPTIMIZER = 'sgd'
    # Learning rate alpha
    LEARNING_RATE = 0.01
    # Number of steps the Learning rate should (linearly)
    # decay to MIN_LEARNING_RATE
    LEARNING_RATE_DECAY = 800
    # Minimal Learning Rate
    MIN_LEARNING_RATE = 0.00001
    # Momentum
    MOMENTUM = 0.9
    # Clipnorm
    CLIPNORM = 0
    # Clipvalue
    CLIPVALUE = 0


def main():
    params = PARAMETERS
    dom_opt = MNIST_DOMAIN_OPTIONS
    Experiment(params, dom_opt, "results.json", "network.h5")

if __name__ == '__main__':
    main()
