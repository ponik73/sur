import matplotlib.pyplot as plt
from utilities import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_test_score(ll_target, ll_non_target):
    return sum(ll_target) - sum(ll_non_target)

def evaluate_test_data(test_data, 
                       Ws_target, MUs_target, COVs_target, 
                       Ws_non_target, MUs_non_target, COVs_non_target):
    score=[]
    for i in range(len(test_data)):
        tst = test_data[i]
        ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
        ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)
        score.append(calculate_test_score(ll_target, ll_non_target))
    return score

def simulate_run(score_borders, iterations):
    _, train_target = wav16khz2mfcc('../data/target_train', augment=True)
    _, train_non_target = wav16khz2mfcc('../data/non_target_train', augment=True)
    _, test_target = wav16khz2mfcc('../data/target_dev')
    _, test_non_target = wav16khz2mfcc('../data/non_target_dev')

    train_target = np.vstack(train_target)
    train_non_target = np.vstack(train_non_target)

    # Train and test with GMM models with diagonal covariance matrices
    # Decide for number of gaussian mixture components used for the target and non target models
    M_target = 15
    M_non_target = 15

    # Initialize all variance vectors (diagonals of the full covariance matrices) to
    # the same variance vector computed using all the data from the given class
    COVs_target = [np.var(train_target, axis=0)] * M_target
    COVs_non_target = [np.var(train_non_target, axis=0)] * M_non_target

    # Use uniform distribution as initial guess for the weights
    Ws_target = np.ones(M_target) / M_target
    Ws_non_target = np.ones(M_non_target) / M_non_target

    max_avg_correctness = np.loadtxt('max_avg_correctness.txt')
    for iter in range(iterations):
        print("Run:", iter+1)
        # Initialize mean vectors to randomly selected data points from corresponding class
        MUs_target = train_target[randint(1, len(train_target), M_target)]

        # Initialize parameters of non target model
        MUs_non_target = train_non_target[randint(1, len(train_non_target), M_non_target)]

        start_time = time.time()
        # Run n iterations of EM agorithm to train the two GMMs from target and non target
        for jj in range(30):
            [Ws_target, MUs_target, COVs_target, TTL_target] = train_gmm(train_target, Ws_target, MUs_target, COVs_target)
            [Ws_non_target, MUs_non_target, COVs_non_target, TTL_non_target] = train_gmm(train_non_target, Ws_non_target, MUs_non_target, COVs_non_target)
            if (jj+1) % 5 == 0:
                print('Iteration:', jj, ' Total log-likelihoods:', TTL_target, 'for target;', TTL_non_target, 'for non targets.')
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", round(execution_time),"seconds.")

        # Now run recognition for all target test utterances
        score_target = evaluate_test_data(test_target, 
                                    Ws_target, MUs_target, COVs_target, 
                                    Ws_non_target, MUs_non_target, COVs_non_target)

        score_non_target = evaluate_test_data(test_non_target, 
                                    Ws_target, MUs_target, COVs_target, 
                                    Ws_non_target, MUs_non_target, COVs_non_target)
        
        for border in score_borders:
            correct_targets = sum(1 for s in score_target if s > border)
            correct_non_targets = sum(1 for s in score_non_target if s < border)
            total_target = len(score_target)
            total_non_target = len(score_non_target)
            crc_target = correct_targets / total_target * 100
            crc_non_target = correct_non_targets / total_non_target * 100
            crc_avg = (crc_target + crc_non_target) / 2
            print("Border:", border,"avg correctness: {:.2f}".format(crc_avg))
            if crc_avg > max_avg_correctness:
                print("Saving new training parameters with border:", border,"and new max avg correctness: {:.2f}".format(crc_avg))
                # Save training parameters
                np.savetxt('Ws_target.txt', Ws_target)
                np.savetxt('Ws_non_target.txt', Ws_non_target)
                np.savetxt('MUs_target.txt', MUs_target)
                np.savetxt('MUs_non_target.txt', MUs_non_target)
                np.savetxt('COVs_target.txt', COVs_target)
                np.savetxt('COVs_non_target.txt', COVs_non_target)
                np.savetxt('border.txt', [border])
                np.savetxt('max_avg_correctness.txt', [crc_avg])
                max_avg_correctness = crc_avg
            print()

def main():
    # Try borders -400, -350, ... 700, 750
    score_borders = list(range(-400, 800, 50))
    
    # Run n iterations of simulation to get distinct correctnesses
    num_iterations = 10
    simulate_run(score_borders, num_iterations)

main()
