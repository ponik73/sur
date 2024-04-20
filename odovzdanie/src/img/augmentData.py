from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import shutil
import cv2
import numpy as np

PATH_TARGET_TRAIN = "../data/target_train"
PATH_TARGET_DEV = "../data/target_dev"
PATH_NON_TARGET_TRAIN = "../data/non_target_train"
PATH_NON_TARGET_DEV = "../data/non_target_dev"
PATH_DATASET_TARGET_TRAIN = "data/train/target"
PATH_DATASET_TARGET_VALIDATION = "data/validation/target"
PATH_DATASET_NON_TARGET_TRAIN = "data/train/nonTarget"
PATH_DATASET_NON_TARGET_VALIDATION = "data/validation/nonTarget"

def createStructure(src, dst):
    allFiles = os.listdir(src)
    imgFiles = [f for f in allFiles if f.endswith(".png")]

    os.makedirs(dst)

    for filename in imgFiles:
        shutil.copy(os.path.join(src, filename), os.path.join(dst, filename))

def loadImages(dataDir):
    """Loads images from a directory and assigns labels."""
    images = []

    allFiles = os.listdir(dataDir)
    imgFiles = [f for f in allFiles if f.endswith(".png")]

    for filename in imgFiles:
        img = cv2.imread(os.path.join(dataDir, filename))
        img = cv2.resize(img, (80, 80))
        images.append(img)
    
    return np.array(images)

def augmentImages(src, datagen, dst):
    images = loadImages(src)
    targetLen = len(images) * 10

    i = 0
    for batch in datagen.flow(images, batch_size=1, save_to_dir=dst, save_format='png'):
        i += 1
        if i > targetLen:
            break

if __name__ == "__main__":

    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
    createStructure(PATH_TARGET_TRAIN, PATH_DATASET_TARGET_TRAIN)
    createStructure(PATH_TARGET_DEV, PATH_DATASET_TARGET_VALIDATION)
    createStructure(PATH_NON_TARGET_TRAIN, PATH_DATASET_NON_TARGET_TRAIN)
    createStructure(PATH_NON_TARGET_DEV, PATH_DATASET_NON_TARGET_VALIDATION)

    augmentImages(PATH_TARGET_TRAIN, datagen, PATH_DATASET_TARGET_TRAIN)
    augmentImages(PATH_TARGET_DEV, datagen, PATH_DATASET_TARGET_VALIDATION)
    augmentImages(PATH_NON_TARGET_TRAIN, datagen, PATH_DATASET_NON_TARGET_TRAIN)
    augmentImages(PATH_NON_TARGET_DEV, datagen, PATH_DATASET_NON_TARGET_VALIDATION)
