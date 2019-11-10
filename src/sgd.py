import matplotlib
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os


matplotlib.use("Agg")  # figures can be saved in the background
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


def load_data(image_paths):
    data = []
    labels = []

    random.seed(42)
    random.shuffle(image_paths)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        label = image_path.split(os.path.sep)[-2]
        labels.append(1 if label == 'faces' else 0)
    return data, labels


def create_plot(epochs, H):
    """Plot the training loss and accuracy"""

    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["plot"])


def main(init_lr: float = 0.01, epochs: int = 75):
    print("Loading images...")
    image_paths = sorted(list(paths.list_images(args["dataset"])))
    data, labels = load_data(image_paths)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    (X_train, X_test, y_train, y_test) = train_test_split(
        data, labels, test_size=0.25, random_state=42)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(y_train.shape[1], activation="softmax"))

    print("Training network...")
    opt = SGD(lr=init_lr)
    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    H = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=32)

    print("Evaluating network...")
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(
        y_test.argmax(axis=1), predictions.argmax(axis=1),
        target_names=['cars', 'faces'])
    )
    create_plot(epochs, H)


main()
