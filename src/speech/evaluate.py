import numpy as np
from utilities import load_file, logpdf_gmm

def get_file_name_from_path(path):
    # Get the last part after last slash
    last_part = path.split("/")[-1]

    # Remove the file extension
    file_name = last_part.split(".")[0]

    return file_name

def evaluate_directory(test_data):
    # Load Ws_target and Ws_non_target
    Ws_target = np.loadtxt('Ws_target.txt')
    Ws_non_target = np.load('Ws_non_target.txt')

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
        score_res = sum(ll_target) - sum(ll_non_target)
        score.append(score_res)
    return score

def print_score_results(files_names, score, decision_score_border):
    for i in range(len(files_names)):
        print(get_file_name_from_path(files_names[i]), end=' ')
        print(score[i], end=' ')
        if score[i] > decision_score_border:
            print('1')
        else:
            print('0')

def main(dir_name):
    # Load keys and values from .wav files
    files_names, test_data = load_file(dir_name)
    score = evaluate_directory(test_data)
    print_score_results(files_names, score, 200)

main('../data/target_dev')