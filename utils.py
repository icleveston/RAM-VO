import os
import json
import numpy as np
import matplotlib.pyplot as plt
import six
import matplotlib.patches as patches
import torch
from PIL import Image
from prettytable import PrettyTable


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_parameters(model, print_table=False):
    
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    
    if print_table:
        print(table)
        
    print(f"[*] Total Trainable Params: {total_params}")
    
    return total_params

def denormalize(T, coords):
    return 0.5 * ((coords + 1.0) * T)

def denormalize_displacement(l, scale=10):
    return l*scale

def bounding_box(x, y, size, color="w"):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype="float32")
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype="float32")
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert("RGB")
    
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
        
    if view:
        img.show()
        
    x = np.asarray(img, dtype="float32")
    
    if expand:
        x = np.expand_dims(x, axis=0)
        
    x /= 255.0
    
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype("uint8"), "RGB")


def plot_images(images, gd_truth):
    
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2)

    image_0 = images[0][0].T
    image_1 = images[1][0].T

    # plot the image
    axes[0].imshow(image_0, cmap="Greys_r")
    axes[1].imshow(image_1, cmap="Greys_r")

    xlabel = "Displacement: {}".format(str(gd_truth[0]))
    axes[1].set_xlabel(xlabel)
   
    major_ticks = np.arange(0, 100, 5)
    minor_ticks = np.arange(0, 100, 1)

    axes[0].set_xticks(major_ticks)
    axes[0].set_xticks(minor_ticks, minor=True)
    axes[0].set_yticks(major_ticks)
    axes[0].set_yticks(minor_ticks, minor=True)
    axes[1].set_xticks(major_ticks)
    axes[1].set_xticks(minor_ticks, minor=True)
    axes[1].set_yticks(major_ticks)
    axes[1].set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    axes[0].grid(which='both')
    axes[1].grid(which='both')

    # Or if you want different settings for the grids:
    axes[0].grid(which='minor', alpha=0.5)
    axes[0].grid(which='major', alpha=0.8)
    axes[1].grid(which='minor', alpha=0.5)
    axes[1].grid(which='major', alpha=0.8)
    
    axes[0].set_xlim(1, 99)
    axes[0].set_ylim(1, 99)
    axes[1].set_xlim(1, 99)
    axes[1].set_ylim(1, 99)

    plt.show()


def prepare_dirs(config):
    for path in [config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    model_name = "ram_{}_{}x{}_{}".format(
        config.num_glimpses, config.patch_size, config.patch_size, config.glimpse_scale
    )
    filename = model_name + "_params.json"
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, "w") as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def render_table(data, data_dir, filename, col_width=3.0, row_height=0.625, font_size=14,
                    header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                    bbox=[0, 0, 1, 1], header_columns=0,
                    ax=None, **kwargs):
    
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
            
    # Save the table
    plt.savefig(os.path.join(data_dir, filename), orientation='landscape', dpi=80)
