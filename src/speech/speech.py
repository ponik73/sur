import matplotlib.pyplot as plt
from utilities import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint

train_target = wav16khz2mfcc('../data/target_train').values()
train_non_target = wav16khz2mfcc('../data/non_target_train').values()
test_target = wav16khz2mfcc('../data/target_dev').values()
test_non_target = wav16khz2mfcc('../data/non_target_dev').values()

# print('Non target train:')
# print(type(train_non_target))
# print()

train_target = np.vstack(list(train_target))
train_non_target = np.vstack(list(train_non_target))
test_target = list(test_target)
test_non_target = list(test_non_target)
dim = train_target.shape[1]

# PCA reduction to 2 dimensions

cov_tot = np.cov(np.vstack([train_target, train_non_target]).T, bias=True)
# take just 2 largest eigenvalues and corresponding eigenvectors
d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim-2, dim-1))

train_target_pca = train_target.dot(e)
train_non_target_pca = train_non_target.dot(e)
# plt.plot(train_target_pca[:,1], train_target_pca[:,0], 'b.', ms=1)
# plt.plot(train_non_target_pca[:,1], train_non_target_pca[:,0], 'r.', ms=1)
# plt.show()

# LDA reduction to 1 dimenzion (only one LDA dimensionis available for 2 classes)
n_target = len(train_target)
n_non_target = len(train_non_target)
cov_wc = (n_target*np.cov(train_target.T, bias=True) + n_non_target*np.cov(train_non_target.T, bias=True)) / (n_target + n_non_target)
cov_ac = cov_tot - cov_wc
d, e = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim-1, dim-1))
# plt.figure()
# junk = plt.hist(train_target.dot(e), 40, histtype='step', color='b', density=True)
# junk = plt.hist(train_non_target.dot(e), 40, histtype='step', color='r', density=True)
# plt.show()
# Distribution in this single dimensional space are reasonable separated

# Lets define uniform a-priori probabilities of classes:
P_target = 0.5
P_non_target = 1 - P_target

# For one target test utterance (test_target[0]), obtain frame-by-frame log-likelihoods
# with two models, one trained using target and second using non target training data.
# In this case, the models are single gaussians with diagonal covariance matrices.

ll_target = logpdf_gauss(test_target[0], np.mean(train_target, axis=0), np.var(train_target, axis=0))
ll_non_target = logpdf_gauss(test_target[0], np.mean(train_non_target, axis=0), np.var(train_non_target, axis=0))

# Plot the frame-by-frame likelihoods obtained with two models; Note that
# 'll_target' and ''ll_non_target' are log likelihoods, so we need to use
# exp function
# plt.figure()
# plt.plot(np.exp(ll_target), 'b')
# plt.plot(np.exp(ll_non_target), 'r')
# plt.show()

# Plot frame-by-frame posteriors
posterior_target = np.exp(ll_target) * P_target / (np.exp(ll_target)*P_target + np.exp(ll_non_target)*P_non_target)
# Alternatively the posterior can by computed using log odds ratio and logistic sigmoid function as:
# posterior_m = logistic_sigmoid(ll_target - ll_non_target + log(P_target/P_non_target))
# plt.figure()
# plt.plot(posterior_m, 'g')
# plt.plot(1 - posterior_m, 'r')
# plt.show()

# Plot frame-by-frame log-likelihoods
# plt.figure()
# plt.plot(ll_target, 'g')
# plt.plot(ll_non_target, 'r')
# plt.show()

# But, we do not want to make frame-by-frame decisions. We want to recognize the
# whole segment. Applying frame independeny assumption, we sum log-likelihoods.
# We decide for class 'target' if the following quantity is positive.
result = (sum(ll_target) + np.log(P_target)) - (sum(ll_non_target) + np.log(P_non_target))
print(result)

# Repeating the whole excercise, now with gaussian models with full covariance
# matrices

ll_target = logpdf_gauss(test_target[0], *train_gauss(train_target))
ll_non_target = logpdf_gauss(test_target[0], *train_gauss(train_non_target))
# '*' before 'train_gauss' passes both return values (mean and cov) as parameters of 'logpdf_gauss'

posterior_target = np.exp(ll_target)*P_target /(np.exp(ll_target) * P_target + np.exp(ll_non_target) * P_non_target)
# plt.figure(); plt.plot(posterior_target, 'g'); plt.plot(1-posterior_target, 'r')
# plt.figure(); plt.plot(ll_target, 'g'); plt.plot(ll_non_target, 'r')
# plt.show()
print((sum(ll_target) + np.log(P_target)) - (sum(ll_non_target) + np.log(P_non_target))) 

# Now run recognition for all target test utterances
# To do the same for non target set "test_set=test_non_target"
score=[]
mean_target, cov_target = train_gauss(train_target)
mean_non_target, cov_non_target = train_gauss(train_non_target)
for tst in test_target:
    ll_target = logpdf_gauss(tst, mean_target, cov_target)
    ll_non_target = logpdf_gauss(tst, mean_non_target, cov_non_target)
    score.append((sum(ll_target) + np.log(P_target)) - (sum(ll_non_target) + np.log(P_non_target)))
# print(score)

# Run recogniction with 1-dimensional LDA projected data
score=[]
mean_target, cov_target = train_gauss(train_target.dot(e))
mean_non_target, cov_non_target = train_gauss(train_non_target.dot(e))
for tst in test_target:
    ll_target = logpdf_gauss(tst.dot(e), mean_target, np.atleast_2d(cov_target))
    ll_non_target = logpdf_gauss(tst.dot(e), mean_non_target, np.atleast_2d(cov_non_target))
    score.append((sum(ll_target) + np.log(P_target)) - (sum(ll_non_target) + np.log(P_non_target)))
print(score)

# Train and test with GMM models with diagonal covariance matrices
# Decide for number of gaussian mixture components used for the target model
M_target = 5

# Initialize mean vectors, covariance matrices and weights of mixture components
# Initialize mean vectors to randomly selected data points from corresponding class
MUs_target = train_target[randint(1, len(train_target), M_target)]

# Initialize all variance vectors (diagonals of the full covariance matrices) to
# the same variance vector computed using all the data from the given class
COVs_target = [np.var(train_target, axis=0)] * M_target

# Use uniform distribution as initial guess for the weights
Ws_target = np.ones(M_target) / M_target

# Initialize parameters of non target model
M_non_target = 5
MUs_non_target = train_non_target[randint(1, len(train_non_target), M_non_target)]
COVs_non_target = [np.var(train_non_target, axis=0)] * M_non_target
Ws_non_target = np.ones(M_non_target) / M_non_target

# Run 30 iterations of EM agorithm to train the two GMMs from target and non target
for jj in range(30):
    # print('ws:', Ws_target)
    # print('mus:',MUs_target)
    # print('covs:',COVs_target)
    [Ws_target, MUs_target, COVs_target, TTL_target] = train_gmm(train_target, Ws_target, MUs_target, COVs_target)
    [Ws_non_target, MUs_non_target, COVs_non_target, TTL_non_target] = train_gmm(train_non_target, Ws_non_target, MUs_non_target, COVs_non_target)
    # print('Iteration:', jj, ' Total log-likelihoods:', TTL_target, 'for target;', TTL_non_target, 'for non targets.')

# Now run recognition for all target test utterances
# To do the same for non target set "test_set=test_non_target"
score=[]
for tst in test_target:
    ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
    ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)
    score.append((sum(ll_target) + np.log(P_target)) - (sum(ll_non_target) + np.log(P_non_target)))
print(score)

score=[]
for tst in test_non_target:
    ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
    ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)
    score.append((sum(ll_target) + np.log(P_target)) - (sum(ll_non_target) + np.log(P_non_target)))
print(score)
