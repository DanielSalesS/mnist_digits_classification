import os
import numpy as np
import gzip
import urllib.request


def download_file(url, source_path):
    out_file = source_path + url.split("/")[-1]

    # Download archive
    try:
        response = urllib.request.urlopen(url)
        with urllib.request.urlopen(url) as response:
            with open(out_file, 'wb') as f:
                f.write(response.read())

    except Exception as e:
        print(e)


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)

    return images, labels


def data_pipeline(source_path, target_path, urls):

    for url in urls:
        download_file(url, source_path)

    train_images, train_labels = load_mnist(path=source_path, kind='train')
    train_images = train_images.reshape(len(train_labels), 28, 28)
    np.save(target_path + "train_images.npy", train_images)
    np.save(target_path + "train_labels.npy", train_labels)

    test_images, test_labels = load_mnist(path=source_path, kind='t10k')
    test_images = test_images.reshape(len(test_labels), 28, 28)
    np.save(target_path + "test_images.npy", test_images)
    np.save(target_path + "test_labels.npy", test_labels)


if __name__ == "__main__":

    source_path = '../data/raw/'
    target_path = '../data/processed/'
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]

    data_pipeline(source_path, target_path, urls)
