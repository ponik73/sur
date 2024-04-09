import os
import pandas as pd
from model import trainModel
from dataPreprocess import loadData, preprocessData
from tensorflow.keras.models import load_model

PATH_TARGET_TRAIN = "../data/target_train"
PATH_TARGET_DEV = "../data/target_dev"
PATH_NON_TARGET_TRAIN = "../data/non_target_train"
PATH_NON_TARGET_DEV = "../data/non_target_dev"
IMG_WIDTH = IMG_HEIGHT = 224
PATH_MODEL = "one_person_detector.keras"
EPOCHS = 11
BATCH_SIZE = 20
LEARNING_RATE = 0.0023856
PATH_EVAL = "eval"
PATH_OUTPUT = "../image_nn.txt"

if __name__ == "__main__":
  trainData, validationData = preprocessData(PATH_TARGET_TRAIN, PATH_TARGET_DEV, PATH_NON_TARGET_TRAIN, PATH_NON_TARGET_DEV, IMG_WIDTH, IMG_HEIGHT)

  if os.path.isfile(PATH_MODEL):
    model = load_model(PATH_MODEL)
  else:
    model = trainModel(trainData, validationData, IMG_HEIGHT, IMG_WIDTH, EPOCHS, BATCH_SIZE, LEARNING_RATE)

  # print(model.summary())

  # Load eval data and filenames
  evalData, filenames = loadData(PATH_EVAL, None, IMG_WIDTH, IMG_HEIGHT, eval=True)
  df = pd.DataFrame(filenames, columns=["filename"])
  df["filename"] = df["filename"].apply(lambda x: x[:-4])

  df["softPrediction"] = model.predict(evalData)
  df["softPrediction"] = df["softPrediction"].apply(lambda x: round(x, 3))
  df["hardPrediction"] = df["softPrediction"].apply(lambda x: 1 if x > 0.5 else 0)

  with open(PATH_OUTPUT, "wb") as f:
    for _, row in df.iterrows():
        s = f'{row.iloc[0]} {row.iloc[1]} {int(row.iloc[2])}\n'
        f.write(s.encode("ascii"))
