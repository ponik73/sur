from img.image_NN import evaluate
from speech.evaluate import evaluate_speech_data
import pandas as pd
import os

if __name__ == "__main__":
    # Test data directory
    dir_with_test_data = "data/non_target_dev"

    # Img
    pathModel = "img/one_person_detector.keras"
    pathEval = "img/eval"
    dfImg = evaluate(pathModel, pathEval)
    dfImg = dfImg.rename(columns={"softPrediction": "softPredictionImg", "hardPrediction": "hardPredictionImg"})

    # Audio
    dfAudio = evaluate_speech_data(dir_with_test_data, return_probabilities=True)
    
    # Join predictions on filename
    combinedPrediction = pd.merge(dfImg, dfAudio, on="filename")
    
    # Evaluate soft prediction
    combinedPrediction["softPrediction"] = combinedPrediction["softPredictionImg"]*0.5 + combinedPrediction["softPredictionAudio"]*0.5
    combinedPrediction["softPrediction"] = combinedPrediction["softPrediction"].apply(lambda x: round(x, 3))
    
    # Evaluate hard prediction
    combinedPrediction["hardPrediction"] = combinedPrediction["softPrediction"].apply(lambda x: 1 if x > 0.5 else 0)

    # Write out combined prediction summary
    with open("predictions/combined.txt", "wb") as f:
        for _, row in combinedPrediction.iterrows():
            s = f'{row.iloc[0]} {row.iloc[-2]} {int(row.iloc[-1])}\n'
            f.write(s.encode("ascii"))

    # Write out image prediction summary
    with open("predictions/image.txt", "wb") as f:
        for _, row in combinedPrediction.iterrows():
            s = f'{row.iloc[0]} {row.iloc[1]} {int(row.iloc[2])}\n'
            f.write(s.encode("ascii"))

    # Write out speech prediction summary
    with open("predictions/speech.txt", "wb") as f:
        for _, row in combinedPrediction.iterrows():
            s = f'{row.iloc[0]} {row.iloc[3]} {int(row.iloc[4])}\n'
            f.write(s.encode("ascii"))