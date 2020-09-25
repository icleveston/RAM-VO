import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def n_random_crop(img, height, width, n):
    """
    This method takes a image, crops randomly n crops of size height x width and returns the n crops.
	
    Parameters
    ----------
    img: np.array
		The image as numpy array
    height, width: int
		The height and width of the crops
	n: int
		The amount of crops which will be returned
	
	Returns
	----------
	crops: np.array
		n crops of size height x width
    """
    crops = []
    img_width, img_height = img.shape
    for i in range(n):
        x = np.random.randint(0, img_width - width)
        y = np.random.randint(0, img_height - height)
        crops.append(img[x:x + height, y:y + width])
    return np.array(crops)

def get_cluttered_translated_mnist(n, canvas_height, canvas_width, crop_height, crop_width):
    """
    This method creates the cluttered translated mnist described in the Paper Recurrent Models of Visual Attention.
    
    Parameters
    ----------
    n: int
        The amount of clutter which will be added
    canvas_height, canvas_width: int
        The height and width of the canvas - the height and width of the output images
    crop_height, crop_width:
        The height and width of the clutter which will be added
    
    Returns
    ----------
    (X_train, y_train), (X_test, y_test): (np.array, np.array), (np.array, np.array)
        The test and train dataset with shape (-1, canvas_height, canvas_width, 1) and the labels (not one hot encoded)
    """
    
    # load all data, labels are one-hot-encoded, images are flatten and pixel squashed between [0,1]
        
    (train_images, y_train), (test_images, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = np.zeros((train_images.shape[0], canvas_height, canvas_width))
    X_test = np.zeros((test_images.shape[0], canvas_height, canvas_width))

    for i in range(train_images.shape[0]):
        X_train[i] = random_translation(X_train[i], train_images[i])
        indixes = np.where(y_train == y_train[3])[0]
        random_index = np.random.randint(0, len(indixes))
        crops = n_random_crop(train_images[random_index], crop_height, crop_width, n)

        for j in range(n):
            rand_x, rand_y = np.random.randint(0, canvas_height - crop_height), np.random.randint(0, canvas_width - crop_width)
            X_train[i][rand_x:rand_x + crop_height, rand_y:rand_y + crop_width] = crops[j]

    for i in range(test_images.shape[0]):
        X_test[i] = random_translation(X_test[i], test_images[i])
        indixes = np.where(y_test == y_test[i])[0]
        random_index = np.random.randint(0, len(indixes))
        crops = n_random_crop(test_images[random_index], crop_height, crop_width, n)
        for j in range(n):
            rand_x, rand_y =np.random.randint(0, canvas_height - crop_height),  np.random.randint(0, canvas_width - crop_width)
            X_test[i][rand_x:rand_x + crop_height, rand_y:rand_y + crop_width] = crops[j]

    return (X_train, y_train), (X_test, y_test)

def random_translation(canvas, img):
    """
    This method takes a image and canvas and places the image randomly on the canvas.

    Parameters
    ----------
    img: np.array
        The image as numpy array
    canvas: np.array
        A empty image which has a larger size than img (normally)

    Returns
    ----------
    canvas: np.array
        The canvas with the img on it
    """
    canvas_width, canvas_height = canvas.shape
    img_width, img_height = img.shape
    rand_X, rand_Y = np.random.randint(0, canvas_width - img_width), np.random.randint(0, canvas_height - img_height)
    canvas[rand_X:rand_X + img_width, rand_Y:rand_Y + 28] = img
    return np.copy(canvas)

def get_translated_mnist(cancas_height, canvas_width):
    """
    This method creates the translated mnist described in the Paper Recurrent Models of Visual Attention.

    Parameters
    ----------
    canvas_height, canvas_width: int
        The height and width of the canvas - the height and width of the output images

    Returns
    ----------
    (X_train, y_train), (X_test, y_test): (np.array, np.array), (np.array, np.array)
        The test and train dataset with shape (-1, canvas_height, canvas_width, 1) and the labels (not one hot encoded)
    """
    (X_train, train_labels), (X_test, test_labels) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    train_images = np.zeros((X_train.shape[0], cancas_height, canvas_width))
    test_images = np.zeros((X_test.shape[0], cancas_height, canvas_width))

    for i in range(train_images.shape[0]):
        train_images[i] = random_translation(train_images[i], X_train[i])
        
    for i in range(test_images.shape[0]):
        test_images[i] = random_translation(test_images[i], X_test[i])

    return (train_images, train_labels), (test_images, test_labels)

def get_mnist(one_hot_enc, normalized, flatten):
    """
    This method loads the MNIST data and returns it

    Parameters
    ----------
    one_hot_enc: boolean
        If its set then the labels will be returned as one hot encoded vectors
    normalized: boolean
        If its set then the images will be normalized to [0, 1]
    flatten:
        If its set then the images will be flatted out, thus the return shape will be (-1, 28*28).
        Else the shape will be (-1, 28, 28, 1)
        
    Returns
    ----------
    (X_train, y_train), (X_test, y_test): (np.array, np.array), (np.array, np.array)
        The test and train dataset
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)      
    else:
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
    if normalized:
        X_train = (X_train/255).astype(np.float32)
        X_test = (X_test/255).astype(np.float32)
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
    if one_hot_enc:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)
            
def minibatcher(inputs, targets, batchsize, shuffle=False):
    """
    This method creates a iterable batcher

    Parameters
    ----------
    inputs, targets: np.arrays
        The input (e.g. training images) and targets (e.g. training labels) which the minibatcher should make batches of
    batchsize: int
        The size of a single batch
    shuffle: boolean (default False)
        If its true then the batches will be random
        
    Returns
    ----------
    minibatcher: iterable
        In order to use it you'll need to iterate though the minibatcher e.g.:
        
        batcher = minibatcher(X, y, 200, True)
        for X_batch, y_batch in batcher:
            print(X_batch.shape) -> (batchsize, ...)
            print(y_batch.shape) -> (batchsize, ...)
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
