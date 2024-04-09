import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def trainModel(trainData, validationData, imgWidth, imgHeight, epochs, batchSize, learningRate):
    model = Sequential([
      Conv2D(176, (3, 3), activation="relu", input_shape=(imgHeight, imgWidth, 3)),
      MaxPooling2D((2, 2)),
      Conv2D(256, (3, 3), activation="relu"),
      MaxPooling2D((2, 2)),
      Flatten(),
      Dense(192, activation="relu"),
      Dense(32, activation="relu"),
      Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learningRate), metrics=["accuracy"])
    model.fit(trainData, validation_data=validationData, epochs=epochs, batch_size=batchSize, steps_per_epoch=(len(trainData) // batchSize))
    model.save("one_person_detector.keras")

    return model
