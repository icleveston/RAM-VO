import tf_mnist_loader
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MNIST():
    """
    Class for downloading and preprocessing the MNIST image dataset
    The Code for creating the glimpses is based on https://github.com/jlindsey15/RAM
    """

    def __init__(self, mnist_size, batch_size, channels, scaling, sensorBandwidth, depth, loc_std, unit_pixels, translate, translated_mnist_size):

        self.mnist_size = mnist_size
        self.batch_size = batch_size
        self.channels = channels # grayscale
        self.scaling = scaling # zooms -> scaling * 2**<depth_level>
        self.sensorBandwidth = sensorBandwidth # fixed resolution of sensor
        self.sensorArea = self.sensorBandwidth**2
        self.depth = depth # zooms
        self.unit_pixels = unit_pixels
        self.dataset = tf_mnist_loader.read_data_sets("mnist_data")

        self.loc_std = loc_std # std when setting the location

        self.translate = translate
        if translate:
            self.translated_mnist_size = mnist_size
            self.mnist_size = translated_mnist_size

    def get_batch_train(self, batch_size):
        X, Y = self.dataset.train.next_batch(batch_size)
        if self.translate:
           X, _ = self.convertTranslated(X, self.translated_mnist_size, self.mnist_size)
        return X,Y

    def get_batch_test(self, batch_size):
        X, Y = self.dataset.test.next_batch(batch_size)
        if self.translate:
            X, _ = self.convertTranslated(X, self.translated_mnist_size, self.mnist_size)
        return X,Y

    def get_batch_validation(self, batch_size):
        X, Y = self.dataset.validation.next_batch(batch_size)
        if self.translate:
            X, _ = self.convertTranslated(X, self.translated_mnist_size, self.mnist_size)
        return X,Y

    def glimpseSensor(self, img, normLoc):
        assert not np.any(np.isnan(normLoc))," Locations have to be between 1, -1: {}".format(normLoc)
        assert np.any(np.abs(normLoc)<=1)," Locations have to be between 1, -1: {}".format(normLoc)

        loc = normLoc * (self.unit_pixels * 2.)/ self.mnist_size # normLoc coordinates are between -1 and 1
        # Convert location [-1,1] into MNIST Coordinates:
        loc = np.around(((loc + 1) / 2.0) * self.mnist_size)

        loc = loc.astype(np.int32)

        img = np.reshape(img, (self.batch_size, self.mnist_size, self.mnist_size, self.channels))

        zooms = []

        # process each image individually
        for k in xrange(self.batch_size):
            imgZooms = []
            one_img = img[k,:,:,:]
            offset = self.sensorBandwidth* (self.scaling ** (self.depth-1))

            # pad image with zeros
            one_img = self.pad_to_bounding_box(one_img, offset, offset, \
                offset + self.mnist_size, offset + self.mnist_size)

            for i in xrange(self.depth):
                d = int(self.sensorBandwidth * (self.scaling ** i))
                r = d/2

                loc_k = loc[k,:]
                adjusted_loc = offset + loc_k - r

                one_img2 = np.reshape(one_img, (one_img.shape[0],\
                    one_img.shape[1]))

                # crop image to (d x d)
                zoom = one_img2[adjusted_loc[0]:adjusted_loc[0]+d, adjusted_loc[1]:adjusted_loc[1]+d]
                assert not np.any(np.equal(zoom.shape, (0,0))), "Picture has size 0, location {}, depth {}".format(adjusted_loc, d)

                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                zoom = cv2.resize(zoom, (self.sensorBandwidth, self.sensorBandwidth),
                          interpolation=cv2.INTER_LINEAR)
                zoom = np.reshape(zoom, (self.sensorBandwidth, self.sensorBandwidth))
                imgZooms.append(zoom)

            zooms.append(np.stack(imgZooms))

        zooms = np.stack(zooms)

        return zooms

    def pad_to_bounding_box(self, image, offset_height, offset_width, target_height,
                            target_width):
        """Pad `image` with zeros to the specified `height` and `width`.
        Adds `offset_height` rows of zeros on top, `offset_width` columns of
        zeros on the left, and then pads the image on the bottom and right
        with zeros until it has dimensions `target_height`, `target_width`.
        This op does nothing if `offset_*` is zero and the image already has size
        `target_height` by `target_width`.
        Args:
          image: 4-D Tensor of shape `[batch, height, width, channels]` or
                 3-D Tensor of shape `[height, width, channels]`.
          offset_height: Number of rows of zeros to add on top.
          offset_width: Number of columns of zeros to add on the left.
          target_height: Height of output image.
          target_width: Width of output image.
        Returns:
          If `image` was 4-D, a 4-D float Tensor of shape
          `[batch, target_height, target_width, channels]`
          If `image` was 3-D, a 3-D float Tensor of shape
          `[target_height, target_width, channels]`
        Raises:
          ValueError: If the shape of `image` is incompatible with the `offset_*` or
            `target_*` arguments, or either `offset_height` or `offset_width` is
            negative.
        """

        is_batch = True
        image_shape = image.shape
        if image.ndim == 3:
            is_batch = False
            image = np.expand_dims(image, 0)
        elif image.ndim is None:
            is_batch = False
            image = np.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError('\'image\' must have either 3 or 4 dimensions.')

        batch = len(image)
        height = len(image[0])
        width = len(image[0,0])
        depth = len(image[0,0,0])

        after_padding_width = target_width - offset_width - width
        after_padding_height = target_height - offset_height - height

        assert offset_height >= 0, 'offset_height must be >= 0'
        assert offset_width >= 0, 'offset_width must be >= 0'
        assert after_padding_width >= 0, 'width must be <= target - offset'
        assert after_padding_height >= 0, 'height must be <= target - offset'

        # Do not pad on the depth dimensions.
        paddings = np.reshape(
               np.stack([
                0, 0, offset_height, after_padding_height, offset_width,
                after_padding_width, 0, 0
            ]), [4, 2])
        padded = np.pad(image, paddings, 'constant', constant_values=0)

        padded_shape = [i for i in [batch, target_height, target_width, depth]]
        np.reshape(padded, padded_shape)

        if not is_batch:
            padded = np.squeeze(padded, axis=0)

        return padded

    def convertTranslated(self, images, initImgSize, finalImgSize):
        size_diff = finalImgSize - initImgSize
        newimages = np.zeros([self.batch_size, finalImgSize*finalImgSize])
        imgCoord = np.zeros([self.batch_size,2])
        for k in xrange(self.batch_size):
            image = images[k, :]
            image = np.reshape(image, (initImgSize, initImgSize))
            # generate and save random coordinates
            randX = np.random.randint(0, size_diff)
            randY = np.random.randint(0, size_diff)
            imgCoord[k,:] = np.array([randX, randY])
            # padding
            image = np.lib.pad(image, ((randX, size_diff - randX), (randY, size_diff - randY)), 'constant', constant_values = (0))
            newimages[k, :] = np.reshape(image, (finalImgSize*finalImgSize))

        return newimages, imgCoord

def main():
    """
    Test script for checking the image preprocessing
    and to print example images
    :return:
    """
    #Standard MNIST
    #mnist = MNIST(28, 4, 1, 2, 8, 1, 0.11 ,13 ,False , 60)

    #Translated MNIST
    mnist = MNIST(28, 4, 1, 2, 12, 3, 0.11, 26, True, 60)

    mnist_size = mnist.mnist_size
    batch_size = mnist.batch_size
    channels = 1 # grayscale
    glimpses = 4

    save = False
    X, Y= mnist.get_batch_test(batch_size)


    img = np.reshape(X, (batch_size, mnist_size, mnist_size, channels))
    plt.ion()
    plt.show()

    zooms = [mnist.glimpseSensor(X, np.random.uniform(-1, 1,(batch_size, 2))) for x in range(glimpses)]

    for k in xrange(batch_size):
        one_img = img[k,:,:,:]

        plt.title(Y[k], fontsize=40)
        plt.imshow(one_img[:,:,0], cmap=plt.get_cmap('gray'),
                   interpolation="nearest")
        plt.draw()
        #time.sleep(0.05)
        if save:
            plt.savefig("letter_" + repr(k) + ".png")
        plt.pause(1.0001)

        ng = 1
        nz = 1
        for g in zooms:
            #for z in zooms[k]:
            for z in g[k]:
                plt.imshow(z[:,:], cmap=plt.get_cmap('gray'),
                       interpolation="nearest")

                plt.draw()
                if save:
                    plt.savefig("letter_" + repr(k) +
                                "glimpse_" + repr(ng) +
                                "zoom_" + repr(nz) + ".png")
                #time.sleep(0.05)
                plt.pause(1.0001)
                nz += 1
            ng += 1


if __name__ == '__main__':
    main()
