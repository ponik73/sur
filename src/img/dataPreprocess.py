import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def loadData(dataDir, targetLabel, width, height, eval=False):
    """Loads images from a directory and assigns labels."""
    images = []

    allFiles = os.listdir(dataDir)
    imgFiles = [f for f in allFiles if f.endswith(".png")]

    for filename in imgFiles:
        img = cv2.imread(os.path.join(dataDir, filename))
        img = cv2.resize(img, (width, height))
        img = img.astype("float32") / 255.0
        images.append(img)

    if eval:
        return np.array(images), imgFiles
    
    return np.array(images), np.array([targetLabel] * len(images))

def preprocessData(pathTargetTrain, pathTargetDev, pathNonTargetTrain, pathNonTargetDev, width, height):
    """Load and preprocess data"""
    train_x, train_y = [], []
    for path, label in [(pathTargetTrain, 1), (pathNonTargetTrain, 0)]:
        x, y = loadData(path, label, width, height)
        train_x.extend(x)
        train_y.extend(y)

    train_x, train_y = np.array(train_x), np.array(train_y)

    dev_x, dev_y = [], []
    for path, label in [(pathTargetDev, 1), (pathNonTargetDev, 0)]:
        x, y = loadData(path, label, width, height)
        dev_x.extend(x)
        dev_y.extend(y)

    dev_x, dev_y = np.array(dev_x), np.array(dev_y)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    trainDatagen = datagen.flow(train_x, train_y, batch_size=32)

    return trainDatagen, (dev_x, dev_y)