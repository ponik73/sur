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

    # 2. Audio
    #   .
    #   .
    #   .
    #   X. Store as Dataframe
    # dfAudio = pd.DataFrame(dfImg['filename'])
    # dfAudio["softPredictionAudio"] = 0.5
    # dfAudio["hardPredictionAudio"] = 0
    dfAudio = evaluate_speech_data(pathEval, return_probabilities=True)
    # print(dfAudio.head())
    
    # Join predictions on filename
    combinedPrediction = pd.merge(dfImg, dfAudio, on="filename")
    
    # Evaluate soft prediction
    combinedPrediction["softPrediction"] = combinedPrediction["softPredictionImg"]*0.5 + combinedPrediction["softPredictionAudio"]*0.5
    combinedPrediction["softPrediction"] = combinedPrediction["softPrediction"].apply(lambda x: round(x, 3))
    
    # Evaluate hard prediction
    combinedPrediction["hardPrediction"] = combinedPrediction["softPrediction"].apply(lambda x: 1 if x > 0.5 else 0)

    # Write out summary
    with open("predictions/combined.txt", "wb") as f:
        for _, row in combinedPrediction.iterrows():
            s = f'{row.iloc[0]} {row.iloc[-2]} {int(row.iloc[-1])}\n'
            f.write(s.encode("ascii"))