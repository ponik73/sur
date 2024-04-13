import os
import numpy as np
from utilities import wav16khz2mfcc, logpdf_gmm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_file_name_from_path(path):
    # Get the last part after the last path separator
    last_part = os.path.basename(path)

    # Remove the file extension
    file_name = os.path.splitext(last_part)[0]

    return file_name

def evaluate_test_data(test_data, return_probabilities=False):
    # Load Ws_target and Ws_non_target
    Ws_target = np.loadtxt('Ws_target.txt')
    Ws_non_target = np.loadtxt('Ws_non_target.txt')

    # Load MUs_target and MUs_non_target
    MUs_target = np.loadtxt('MUs_target.txt')
    MUs_non_target = np.loadtxt('MUs_non_target.txt')

    # Load COVs_target and COVs_non_target
    COVs_target = np.loadtxt('COVs_target.txt')
    COVs_non_target = np.loadtxt('COVs_non_target.txt')

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
    return score

def print_score_results(files_names, score, decision_score_border):
    for i in range(len(files_names)):
        print(get_file_name_from_path(files_names[i]), end=' ')
        print(score[i], end=' ')
        if score[i] > decision_score_border:
            print('1')
        else:
            print('0')

def main():
    # Load keys and values from .wav files iniside directory
    files_names, test_data = wav16khz2mfcc('../data/non_target_dev', print_file_info=False)

    # Load best decision border for evaluation parameters
    decision_border = np.loadtxt('border.txt')

    # Get list of scores for every item inside test_data
    score = evaluate_test_data(test_data)
    print_score_results(files_names, score, decision_border)

main()