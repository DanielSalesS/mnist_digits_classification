from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
from .utils import load_data


def run_knn_pipeline(data_path, name, model_path='../models/'):

    model = KNeighborsClassifier()

    train_images, train_labels, test_images, test_labels = load_data(
        data_path, image_shape=(784,)
    )

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print('fit...')
    model.fit(train_images, train_labels)

    test_acc = model.score(test_images, test_labels)
    print('\nTest accuracy:', test_acc)

    path_out = model_path + f'/{name}/{name}.sav'
    os.makedirs(os.path.dirname(path_out), exist_ok=True)

    pickle.dump(model, open(path_out, 'wb'))
