import sys
from network import RAM
from MNIST_Processing import MNIST
from matplotlib import pyplot as plt
import numpy as np

# This is not a nice way to implement the different configuration scripts...
if len(sys.argv) > 1:
    if sys.argv[1] == 'run_mnist':
        from run_mnist import MNIST_DOMAIN_OPTIONS
        from run_mnist import PARAMETERS
    elif sys.argv[1] == 'run_translated_mnist':
        from run_translated_mnist import MNIST_DOMAIN_OPTIONS
        from run_translated_mnist import PARAMETERS
    else:
        print "Wrong file name for confiuration file!"
        sys.exit(0)
else:
    print "Give Configuration File as additional argument! \n " \
          "E.g. python evaluate.py run_mnist ./model/network.h5"
    sys.exit(0)

save = True

mnist_size = MNIST_DOMAIN_OPTIONS.MNIST_SIZE
channels = MNIST_DOMAIN_OPTIONS.CHANNELS
scaling = MNIST_DOMAIN_OPTIONS.SCALING_FACTOR
sensorResolution = MNIST_DOMAIN_OPTIONS.SENSOR
loc_std = MNIST_DOMAIN_OPTIONS.LOC_STD
nZooms = MNIST_DOMAIN_OPTIONS.DEPTH
nGlimpses = MNIST_DOMAIN_OPTIONS.NGLIMPSES

#Reduce the batch size for evaluatoin
batch_size = PARAMETERS.BATCH_SIZE

totalSensorBandwidth = nZooms * sensorResolution * sensorResolution * channels
mnist = MNIST(mnist_size, batch_size, channels, scaling, sensorResolution,
              nZooms, loc_std, MNIST_DOMAIN_OPTIONS.UNIT_PIXELS,
              MNIST_DOMAIN_OPTIONS.TRANSLATE, MNIST_DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)

ram = RAM(totalSensorBandwidth, batch_size, nGlimpses,
               PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY,
               PARAMETERS.MIN_LEARNING_RATE, MNIST_DOMAIN_OPTIONS.LOC_STD)

ram.big_net(PARAMETERS.OPTIMIZER,PARAMETERS.LEARNING_RATE,PARAMETERS.MOMENTUM,
                 PARAMETERS.CLIPNORM, PARAMETERS.CLIPVALUE)

if len(sys.argv) > 2:
    if ram.load_model('./', sys.argv[2]):
        print("Loaded wights from " + sys.argv[2] + "!")
    else:
        print("Weights from " + sys.argv[2] +
                     " could not be loaded!")
        sys.exit(0)
else:
    print("No weight file provided! New model initialized!")


plt.ion()
plt.show()

if MNIST_DOMAIN_OPTIONS.TRANSLATE:
    mnist_size = MNIST_DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE

X, Y= mnist.get_batch_test(batch_size)
img = np.reshape(X, (batch_size, mnist_size, mnist_size, channels))
for k in xrange(batch_size):
    one_img = img[k,:,:,:]

    plt.title(Y[k], fontsize=40)
    plt.imshow(one_img[:,:,0], cmap=plt.get_cmap('gray'),
               interpolation="nearest")
    plt.draw()
    #time.sleep(0.05)
    if save:
        plt.savefig("symbol_" + repr(k) + ".png")
    plt.pause(1.0001)

loc = np.random.uniform(-1, 1,(batch_size, 2))
sample_loc = np.tanh(np.random.normal(loc, loc_std, loc.shape))
for n in range(nGlimpses):
    zooms = mnist.glimpseSensor(X,sample_loc)
    ng = 1
    for g in range(batch_size):
        nz = 1
        plt.title(Y[g], fontsize=40)
        for z in zooms[g]:
            plt.imshow(z[:,:], cmap=plt.get_cmap('gray'),
                       interpolation="nearest")

            plt.draw()
            if save:
                plt.savefig("symbol_" + repr(g) + "_" +
                            "glimpse_" + repr(n) + "_" +
                            "zoom_" + repr(nz) + ".png")
            #time.sleep(0.05)
            plt.pause(1.0001)
            nz += 1
        ng += 1
    a_prob, loc = ram.choose_action(zooms, sample_loc)
    sample_loc = np.tanh(np.random.normal(loc, loc_std, loc.shape))
ram.reset_states()


