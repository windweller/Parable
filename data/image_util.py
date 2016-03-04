import cPickle as pickle
import numpy as np
import os
import theano
from scipy.misc import imread
from classifier.layers import conv_layer

"""
  This file reads input from tiny_image_net_A

  divide a dataset into batches of equal size
  return a tuple, each element being a list of batches of data

  Inputs:
  - X: training images
  - Y: training labels
  - batch_size: a number denoting the number of images in each batch

  Then we also have image utils such as blur_image (notice it uses Conv net)
"""


def generate_batches(X, Y, batch_size):
    xybatches = []
    size = X.shape[0] / float(batch_size)
    for i in xrange(int(size)):
        indices = np.random.choice(X.shape[0], size=batch_size, replace=True)
        xbatch = X[indices, :, :, :]
        ybatch = Y[indices]
        xytuple = (xbatch, ybatch)
        xybatches.append(xytuple)
    return xybatches


# to call this function use:
# data = load_tiny_imagenet('<path to tiny imagenet directory>', subtract_mean=True)
def load_tiny_imagenet(path, sub_sample=1.0, dtype=np.float32, subtract_mean=True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
    - sub_sample: 1 means we use 100% of the each class, 0 means 0% of the each class
                  This is not sampling on the whole dataset
                  but more on each class
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (3, 64, 64) array giving mean training image
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.iteritems():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print 'loading training data for synset %d / %d' % (i + 1, len(wnids))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        # adding this to cut-off
        num_images = int(round(num_images * sub_sample))

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int32)
        for j, img_file in enumerate(filenames):

            # break loop when cut-off point
            if num_images == j:
                break

            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids], dtype=np.int32)
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test, dtype=np.int32)

    # this is preprocessing the image
    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names,
        'mean_image': mean_image,
    }


def blur_image(X):
    """
    WARNINGS: could be broken!! (because of the conv_net part)

    A very gentle image blurring operation, to be used as a regularizer for image
    generation.

    Inputs:
    - X: Image data of shape (N, 3, H, W)

    Returns:
    - X_blur: Blurred version of X, of shape (N, 3, H, W)
    """
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {'stride': 1, 'pad': 1}
    for i in xrange(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0
    return conv_layer(X, w_blur, b_blur, blur_param)[0]


def preprocess_image(img, mean_img, mean='image'):
    """
    Convert to float, transepose, and subtract mean pixel

    Input:
    - img: (H, W, 3)

    Returns:
    - (1, 3, H, 3)
    """
    if mean == 'image':
        mean = mean_img
    elif mean == 'pixel':
        mean = mean_img.mean(axis=(1, 2), keepdims=True)
    elif mean == 'none':
        mean = 0
    else:
        raise ValueError('mean must be image or pixel or none')
    return img.astype(np.float32).transpose(2, 0, 1)[None] - mean


def deprocess_image(img, mean_img, mean='image', renorm=False):
    """
    Add mean pixel, transpose, and convert to uint8

    Input:
    - (1, 3, H, W) or (3, H, W)

    Returns:
    - (H, W, 3)
    """
    if mean == 'image':
        mean = mean_img
    elif mean == 'pixel':
        mean = mean_img.mean(axis=(1, 2), keepdims=True)
    elif mean == 'none':
        mean = 0
    else:
        raise ValueError('mean must be image or pixel or none')
    if img.ndim == 3:
        img = img[None]
    img = (img + mean)[0].transpose(1, 2, 0)
    if renorm:
        low, high = img.min(), img.max()
        img = 255.0 * (img - low) / (high - low)
    return img.astype(np.uint8)
