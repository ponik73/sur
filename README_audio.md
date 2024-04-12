## SPEECH

### Score probability

```python
ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)
score = sum(ll_target) + np.log(P_target) - (sum(ll_non_target) + np.log(P_non_target))
```

If the score is positive, the model predicts that the tst file is target, otherwise it is not target.

Correctness of target: 100.00%  
Correctness of non-target: 91.67%

### Average probability

```python
ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)

# Compute log-odds ratio
log_odds_ratio = ll_target - ll_non_target

# Apply sigmoid function to get probability score
probability_score = sigmoid(log_odds_ratio)

# Aggregate probabilities across frames
final_probability_score = np.mean(probability_score)    # Average probability
```

If final_probability_score is >= 0.5, it is target file.

Correctness of target: 100.00%  
Correctness of non-target: 88.33%

### Max probability

```python
ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)

# Compute log-odds ratio
log_odds_ratio = ll_target - ll_non_target

# Apply sigmoid function to get probability score
probability_score = sigmoid(log_odds_ratio)

# Aggregate probabilities across frames
final_probability_score = np.max(probability_score)    # Max probability
```

If final_probability_score is >= 0.5, it is target file.

Correctness of target: 100.00%  
Correctness of non-target: 0.00%

This is because in every .wav file there is probability somewhere around 0.99 that the frame is from target file.