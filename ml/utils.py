import numpy as np


def load_data(data_path, image_shape=(28, 28)):
    train_images = np.load(data_path + "train_images.npy")
    train_labels = np.load(data_path + "train_labels.npy")
    test_images = np.load(data_path + "test_images.npy")
    test_labels = np.load(data_path + "test_labels.npy")

    if image_shape == (784,):
        train_images = train_images.reshape(len(train_labels), 784)
        test_images = test_images.reshape(len(test_labels), 784)

    if image_shape == (28, 28, 1):
        train_images = train_images.reshape(len(train_labels), 28, 28, 1)
        test_images = test_images.reshape(len(test_labels), 28, 28, 1)

    return train_images, train_labels, test_images, test_labels
