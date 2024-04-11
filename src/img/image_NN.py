import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import cv2

PATH_MODEL = "one_person_detector.keras"
EPOCHS = 20
BATCH_SIZE = 20
IMG_SIZE = 80
PATH_EVAL = "miso"
PATH_OUTPUT = "../image_nn.txt"

def loadEval(path, imgSize):
    images = []

    allFiles = os.listdir(path)
    imgFiles = [f for f in allFiles if f.endswith(".png")]

    for filename in imgFiles:
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.resize(img, (imgSize, imgSize))
        img = img.astype("float32") / 255.0
        images.append(img)

    return np.array(images), imgFiles

def loadData(batchSize, imgSize):
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
            'data/train', 
            target_size=(imgSize, imgSize),
            batch_size=batchSize,
            class_mode='binary')

    validation_generator = datagen.flow_from_directory(
            'data/validation',
            target_size=(imgSize, imgSize),
            batch_size=batchSize,
            class_mode='binary')
    
    return train_generator, validation_generator

def trainModel(trainData, validationData, imgSize, epochs, batchSize):
    model = Sequential([
      Input((imgSize, imgSize, 3)),

      Conv2D(32, (3, 3), activation="relu"),
      MaxPooling2D((2, 2)),

      Conv2D(32, (3, 3), activation="relu"), ##
      MaxPooling2D((2, 2)),

      Conv2D(64, (3, 3), activation="relu"),
      MaxPooling2D((2, 2)),

      Flatten(),
      Dense(64, activation="relu"),
      Dropout(0.5),
      Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=["accuracy"])
    model.fit(
      trainData,
      # steps_per_epoch=nb_train_samples // batch_size,
      epochs=epochs,
      validation_data=validationData)#,
      # validation_steps=nb_validation_samples // batch_size)
    model.save("one_person_detector.keras")

    return model


if __name__ == "__main__":
  if os.path.isfile(PATH_MODEL):
    model = load_model(PATH_MODEL)
  else:
    trainGen, validationGen = loadData(BATCH_SIZE, IMG_SIZE)
    model = trainModel(trainGen, validationGen, IMG_SIZE, EPOCHS, BATCH_SIZE)
  
  evalData, filenames = loadEval(PATH_EVAL, IMG_SIZE)

  df = pd.DataFrame(filenames, columns=["filename"])
  df["filename"] = df["filename"].apply(lambda x: x[:-4])

  df["softPrediction"] = model.predict(evalData)
  df["softPrediction"] = df["softPrediction"].apply(lambda x: round(x, 3))
  df["hardPrediction"] = df["softPrediction"].apply(lambda x: 1 if x > 0.5 else 0)

  with open(PATH_OUTPUT, "wb") as f:
    for _, row in df.iterrows():
        s = f'{row.iloc[0]} {row.iloc[1]} {int(row.iloc[2])}\n'
        f.write(s.encode("ascii"))
