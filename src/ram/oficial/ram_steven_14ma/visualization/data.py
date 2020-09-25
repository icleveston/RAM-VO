import matplotlib.pyplot as plt
import numpy as np

def plot_mnist_images(data, label, examples_each_row):
    """
    This methods takes the MNIST images and labels and plots n (examples_each_row) images of each class

    Parameters
    ----------
    data: np.array
        The images of MNIST - a array of shape (n, h, w)
    labels: np.array
        The labels of the images of MNIST/the labels of data. They should not be one hot encoded!
    examples_each_row: int
        The amount of examples which will be plotted for each class
    """
    plt.figure(figsize=(20, 20))
    num_classes = 10

    for c in range(num_classes):
        # Select samples_per_class random keys of the labels == current class
        keys = np.random.choice(np.where(label == c)[0], examples_each_row)
        images = data[keys]
        labels = label[keys]
        for i in range(examples_each_row):
            f = plt.subplot(examples_each_row, num_classes, i * num_classes + c + 1)
            f.axis('off')
            plt.imshow(images[i], cmap='gray',  interpolation='nearest')
            plt.title(labels[i])
            
