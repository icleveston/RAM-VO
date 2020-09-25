#!/usr/bin/env python
import os
import gzip
import struct
from datetime import datetime
import numpy as np 

_RNG_SEED = None

def get_rng(obj=None):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.

    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)

    def preprocess_im(im):
        """ normalize input image to [0., 1.] """
        im = im.astype(np.float32)
        im = im / 255.
        return im


class MNIST:
    """ class for MNIST dataflow

        To access the data of mini-batch, first get data of all the channels
        through batch_data = MNISTData.next_batch_dict()
        then use corresponding key to get label or image through
        batch_data[key].
    """
    def __init__(self, name, data_dir='mnist_data/', batch_size=128, n_use_label=None, n_use_sample=None,
                 batch_dict_name=None, shuffle=True):
        """
        Args:
            name (str): name of data to be read (['train', 'test', 'val'])
            data_dir (str): directory of MNIST data
            n_use_label (int): number of labels to be used
            n_use_sample (int): number of samples to be used
            batch_dict_name (list of str): list of keys for 
                image and label of batch data
            shuffle (bool): whether shuffle data or not
            pf: pre-process function for image data
        """
        assert os.path.isdir(data_dir)
        self._data_dir = data_dir

        self._shuffle = shuffle

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        assert name in ['train', 'test', 'val']
        self._setup(epoch_val=0, batch_size=batch_size)

        self._load_files(name, n_use_label, n_use_sample)
        self._image_id = 0

    def next_batch_dict(self):
        batch_data = self.next_batch()
        data_dict = {key: data for key, data
                     in zip(self._batch_dict_name, batch_data)}
        return data_dict

    def _load_files(self, name, n_use_label, n_use_sample):
        if name == 'train':
            image_name = 'train-images-idx3-ubyte.gz'
            label_name = 'train-labels-idx1-ubyte.gz'
        else:
            image_name = 't10k-images-idx3-ubyte.gz'
            label_name = 't10k-labels-idx1-ubyte.gz'

        image_path = os.path.join(self._data_dir, image_name)
        label_path = os.path.join(self._data_dir, label_name)

        with gzip.open(label_path) as f:
            magic = struct.unpack('>I', f.read(4))
            if magic[0] != 2049:
                raise Exception('Invalid file: unexpected magic number.')
            n_label = struct.unpack('>I', f.read(4))
            label_list = np.fromstring(f.read(n_label[0]), dtype = np.uint8)

        with gzip.open(image_path) as f:
            magic = struct.unpack('>I', f.read(4))
            if magic[0] != 2051:
                raise Exception('Invalid file: unexpected magic number.')
            n_im, rows, cols = struct.unpack('>III', f.read(12))
            image_list = np.fromstring(
                f.read(n_im * rows * cols), dtype = np.uint8)
            image_list = np.reshape(image_list, (n_im, rows, cols, 1))
            im_list = []
            if n_use_sample is not None and n_use_sample < len(label_list):
                remain_sample = n_use_sample // 10 * 10
                left_sample = n_use_sample - remain_sample
                keep_sign = [0 for i in range(10)]
                data_idx = 0
                new_label_list = []
                for idx, im in enumerate(image_list):

                    if remain_sample > 0:
                        if keep_sign[label_list[idx]] < (n_use_sample // 10):
                            keep_sign[label_list[idx]] += 1
                            im_list.append(im)
                            new_label_list.append(label_list[idx])
                            remain_sample -= 1
                    else:
                        break
                im_list.extend(image_list[idx:idx + left_sample])
                new_label_list.extend(label_list[idx:idx + left_sample])
                label_list = new_label_list

            else:
                for im in image_list:
                    im_list.append(im)

        self.im_list = np.array(im_list)
        self.label_list = np.array(label_list)

        if n_use_label is not None and n_use_label < self.size():
            remain_sample = n_use_label // 10 * 10
            left_sample = n_use_label - remain_sample
            keep_sign = [0 for i in range(10)]
            data_idx = 0
            while remain_sample > 0:
                if keep_sign[self.label_list[data_idx]] < (n_use_label // 10):
                    keep_sign[self.label_list[data_idx]] += 1
                    remain_sample -= 1
                else:
                    self.label_list[data_idx] = 10
                data_idx += 1

            self.label_list[data_idx + left_sample:] = 10
        self._suffle_files()

    def _suffle_files(self):
        if self._shuffle:
            idxs = np.arange(self.size())

            self.rng.shuffle(idxs)
            self.im_list = self.im_list[idxs]
            self.label_list = self.label_list[idxs]

    def size(self):
        return self.im_list.shape[0]

    def next_batch(self):
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())
        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_files = self.im_list[start:end]
        batch_label = self.label_list[start:end]

        if self._image_id + self._batch_size > self.size():
            self._epochs_completed += 1
            self._image_id = 0
            self._suffle_files()
        return [batch_files, batch_label]

    def _setup(self, epoch_val, batch_size, **kwargs):
        self._epochs_completed = epoch_val
        self._batch_size = batch_size
        self.rng = get_rng(self)
        try:
            self._suffle_files()
        except AttributeError:
            pass

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed
