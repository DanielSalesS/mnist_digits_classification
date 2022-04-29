from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from .utils import load_data


def run_pipeline_01(data_path, model_name, model_path='../models/'):

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax'),
        ]
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    train_images, train_labels, test_images, test_labels = load_data(data_path)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save(model_path + model_name)


def run_pipeline_02(data_path, name, model_path='../models/'):

    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    train_images, train_labels, test_images, test_labels = load_data(
        data_path, image_shape=(28, 28, 1)
    )
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model.fit(train_images, train_labels, epochs=10, batch_size=32)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save(model_path + name)
