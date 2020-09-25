import matplotlib.pyplot as plt
import numpy as np

def draw_bounding_boxes(img, loc, glimpse_size):
    """
    This method takes a image and location and returns a bounding box of size glimpse_size x glimpse_size

    Parameters
    ----------
    img: np.array
        The image as np.array on which the bounding box will be drawn
    loc: np.array
        The normalized location of the bounding box center e.g. loc=[0.0, 0.0] means that the bounding box is at the center of the image
    glimpse_size: int
        The size of the bounding box/patch which will be added later on

    Returns
    ----------
    patches: matplotlib.patches
        a patch which is the bounding box with size glimpse_size x glimpse_size at location loc on image img
    """
    import matplotlib.patches as patches
    h, w = img.shape
    # draw bounding box fist    
    loc = loc + 1 # making them in the range of [0, 2]
    x_center, y_center = h / 2 * loc[0], w / 2 * loc[1]
    x, y = x_center - glimpse_size/2, y_center - glimpse_size/2
    return patches.Rectangle((x,y),glimpse_size,glimpse_size,linewidth=1,edgecolor='r',facecolor='none')

def plot_prediction_path(image, locs, n_glimpses, glimpse_size):
    """
    This model draws visualizes the locations on the images alias it draws the path of the Recurrent Attention Model which it took in order to predict the label of the image image

    Parameters
    ----------
    image: np.array
        The image on which the prediction was made
    locs: np.array of type float32
        The normalized coordinates of the center of the bounding box which will be drawed. It should have the shape (-1, 2).
    n_glimpses: int
        The amount of glimpses which will be drawn on the image
    glimpse_size: int
        The smallest size (height and width) of every glimpse
    """
    image_h, image_w, image_c = image.shape
    img = image.reshape(image_h, image_w)
    
    max_size = glimpse_size * (2 ** (n_glimpses - 1))
    padding = max_size - glimpse_size
    image = np.pad(img, padding, mode='constant', constant_values=0)
    
    fig, axs = plt.subplots(1, len(locs), figsize=(15, 15))
    axs.ravel()
    for i, loc in enumerate(locs):
        axs[i].imshow(img, cmap='gray',  interpolation='nearest')
        axs[i].set_title(str(i) + ":" + str(loc))
        axs[i].axis("off")
        
        for j in range(n_glimpses):
            current_size = int(glimpse_size * (2 ** (j)))
            box = draw_bounding_boxes(img, loc, current_size)
            axs[i].add_patch(box)
    plt.show()


def draw_bounding_boxes_3d(img, loc, glimpse_size):
    """
    This method takes a image and location and returns a bounding box of size glimpse_size x glimpse_size

    Parameters
    ----------
    img: np.array
        The image as np.array on which the bounding box will be drawn
    loc: np.array
        The normalized location of the bounding box center e.g. loc=[0.0, 0.0] means that the bounding box is at the center of the image
    glimpse_size: int
        The size of the bounding box/patch which will be added later on

    Returns
    ----------
    patches: matplotlib.patches
        a patch which is the bounding box with size glimpse_size x glimpse_size at location loc on image img
    """
    import matplotlib.patches as patches
    h, w, _ = img.shape
    # draw bounding box fist    
    loc = loc + 1 # making them in the range of [0, 2]
    x_center, y_center = h / 2 * loc[0], w / 2 * loc[1]
    x, y = x_center - glimpse_size/2, y_center - glimpse_size/2
    return patches.Rectangle((x,y),glimpse_size,glimpse_size,linewidth=1,edgecolor='r',facecolor='none')

def plot_prediction_path_3d(image, locs, n_glimpses, glimpse_size):
    import matplotlib.pyplot as plt
    """
    This model draws visualizes the locations on the images alias it draws the path of the Recurrent Attention Model which it took in order to predict the label of the image image

    Parameters
    ----------
    image: np.array
        The image on which the prediction was made
    locs: np.array of type float32
        The normalized coordinates of the center of the bounding box which will be drawed. It should have the shape (-1, 2).
    n_glimpses: int
        The amount of glimpses which will be drawn on the image
    glimpse_size: int
        The smallest size (height and width) of every glimpse
    """
    image_h, image_w, image_c = image.shape
    
    max_size = glimpse_size * (2 ** (n_glimpses - 1))
    img = image
    padding = max_size - glimpse_size
    image = np.pad(image, padding, mode='constant', constant_values=0)
    
    fig, axs = plt.subplots(1, len(locs), figsize=(15, 15))
    axs.ravel()
    for i, loc in enumerate(locs):
        axs[i].imshow(img)
        axs[i].set_title(str(i) + ":" + str(loc))
        axs[i].axis("off")
        
        for j in range(n_glimpses):
            current_size = int(glimpse_size * (2 ** (j)))
            box = draw_bounding_boxes_3d(img, loc, current_size)
            axs[i].add_patch(box)
    plt.show()
