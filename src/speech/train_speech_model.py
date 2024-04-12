import matplotlib.pyplot as plt
from utilities import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint

def get_file_name_from_path(path):
    # Get the last part after last slash
    last_part = path.split("/")[-1]

    # Remove the file extension
    file_name = last_part.split(".")[0]

    return file_name

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_file(path, augment=False):
    res_dict = wav16khz2mfcc(path, augment=augment)
    return list(res_dict.keys()), list(res_dict.values())

def calculate_test_score(ll_target, P_target, ll_non_target, P_non_target):
    return (sum(ll_target) + np.log(P_target)) - (sum(ll_non_target) + np.log(P_non_target))

def evaluate_test_data(test_data, 
                       Ws_target, MUs_target, COVs_target, P_target, 
                       Ws_non_target, MUs_non_target, COVs_non_target, P_non_target):
    score=[]
    for i in range(len(test_data)):
        tst = test_data[i]
        ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
        ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)
        score.append(calculate_test_score(ll_target, P_target, ll_non_target, P_non_target))
    return score

def simulate_run(score_borders, iterations):
    _, train_target = load_file('../data/target_train', augment=True)
    _, train_non_target = load_file('../data/non_target_train', augment=True)
    test_target_keys, test_target = load_file('../data/target_dev')
    test_non_target_keys, test_non_target = load_file('../data/non_target_dev')

    train_target = np.vstack(train_target)
    train_non_target = np.vstack(train_non_target)
    test_target = test_target
    test_non_target = test_non_target

    # Define uniform a-priori probabilities of classes:
    P_target = 0.5
    P_non_target = 1 - P_target

    # Train and test with GMM models with diagonal covariance matrices
    # Decide for number of gaussian mixture components used for the target and non target models
    M_target = 5
    M_non_target = 5

    # Initialize all variance vectors (diagonals of the full covariance matrices) to
    # the same variance vector computed using all the data from the given class
    COVs_target = [np.var(train_target, axis=0)] * M_target
    COVs_non_target = [np.var(train_non_target, axis=0)] * M_non_target

    # Use uniform distribution as initial guess for the weights
    Ws_target = np.ones(M_target) / M_target
    Ws_non_target = np.ones(M_non_target) / M_non_target

    for border in score_borders:
        print("Score border:",border)
        crc_targets=[]
        crc_non_targets=[]
        avg_crcs=[]
        for iter in range(iterations):
            # Initialize mean vectors, covariance matrices and weights of mixture components
            # Initialize mean vectors to randomly selected data points from corresponding class
            MUs_target = train_target[randint(1, len(train_target), M_target)]

            # Initialize parameters of non target model
            MUs_non_target = train_non_target[randint(1, len(train_non_target), M_non_target)]

            # Run 50 iterations of EM agorithm to train the two GMMs from target and non target
            for jj in range(50):
                [Ws_target, MUs_target, COVs_target, TTL_target] = train_gmm(train_target, Ws_target, MUs_target, COVs_target)
                [Ws_non_target, MUs_non_target, COVs_non_target, TTL_non_target] = train_gmm(train_non_target, Ws_non_target, MUs_non_target, COVs_non_target)
                # print('Iteration:', jj, ' Total log-likelihoods:', TTL_target, 'for target;', TTL_non_target, 'for non targets.')

            # Now run recognition for all target test utterances
            score = evaluate_test_data(test_target, 
                                       Ws_target, MUs_target, COVs_target, P_target, 
                                       Ws_non_target, MUs_non_target, COVs_non_target, P_non_target)
            correct = sum(1 for s in score if s > border)
            total = len(score)
            correctness_target = (correct / total) * 100

            score = evaluate_test_data(test_non_target, 
                                       Ws_target, MUs_target, COVs_target, P_target, 
                                       Ws_non_target, MUs_non_target, COVs_non_target, P_non_target)
            correct = sum(1 for s in score if s < border)
            total = len(score)
            correctness_non_target = (correct / total) * 100

            avg_correctness = (correctness_target + correctness_non_target) / 2
            crc_targets.append(correctness_target)
            crc_non_targets.append(correctness_non_target)
            avg_crcs.append(avg_correctness)
            print("Run:", iter+1, ", crc target: {:.2f}".format(correctness_target), ", crc non target: {:.2f}".format(correctness_non_target), ", avg crc: {:.2f}".format(avg_correctness))

        if iterations > 1:
            print("Average target correctness: {:.2f}".format(np.mean(crc_targets)))
            print("Average non target correctness: {:.2f}".format(np.mean(crc_non_targets)))
            print("Average model correctness: {:.2f}".format(np.mean(avg_crcs)))
        print()

    return correctness_target, correctness_non_target

def main():
    score_borders = [-200, 0, 200, 400, 600]
    # score_borders = [400]
    # Run n iterations of simulation to get distinct correctnesses
    num_iterations = 10
    simulate_run(score_borders, num_iterations)

main()
