import os
import numpy as np
import pandas as pd
from speech.utilities import wav16khz2mfcc, logpdf_gmm  # Change to 'utilities' instead of 'speech.utilities' when running from speech directory

# Change this to empty string when running from speech directory
parent_dir = "speech/"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_file_name_from_path(path):
    # Get the last part after the last path separator
    last_part = os.path.basename(path)

    # Remove the file extension
    file_name = os.path.splitext(last_part)[0]

    return file_name

def evaluate_speech_data(dir_name, return_probabilities=False):
    # Load keys and values from .wav files iniside directory
    files_names, test_data = wav16khz2mfcc(dir_name, print_file_info=False)

    # Load Ws_target and Ws_non_target
    Ws_target = np.loadtxt(parent_dir + 'Ws_target.txt')
    Ws_non_target = np.loadtxt(parent_dir + 'Ws_non_target.txt')

    # Load MUs_target and MUs_non_target
    MUs_target = np.loadtxt(parent_dir + 'MUs_target.txt')
    MUs_non_target = np.loadtxt(parent_dir + 'MUs_non_target.txt')

    # Load COVs_target and COVs_non_target
    COVs_target = np.loadtxt(parent_dir + 'COVs_target.txt')
    COVs_non_target = np.loadtxt(parent_dir + 'COVs_non_target.txt')

    # Load best decision border for evaluation parameters
    if not return_probabilities:
        decision_border = np.loadtxt(parent_dir + 'border.txt')
    else:
        decision_border = 0.5

    score=[]
    for i in range(len(test_data)):
        tst = test_data[i]
        ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
        ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)
        if not return_probabilities:
            score_res = sum(ll_target) - sum(ll_non_target)
            score.append(score_res)
        else:
            # Compute log-odds ratio
            log_odds_ratio = ll_target - ll_non_target

            # Apply sigmoid function to get probability score
            probability_score = sigmoid(log_odds_ratio)

            # Aggregate probabilities across frames
            final_probability = np.mean(probability_score)

            score.append(final_probability)

    dfAudio = pd.DataFrame({
        "filename": files_names,
        "softPredictionAudio": score,
        "hardPredictionAudio": [0] * len(score)
    })

    dfAudio["filename"] = dfAudio["filename"].apply(get_file_name_from_path)
    dfAudio["hardPredictionAudio"] = (dfAudio["softPredictionAudio"] > decision_border).astype(int)

    return dfAudio
