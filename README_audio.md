## SPEECH

### How to run the program

`speech.py` file is ready to test different `score_borders` and number of `iterations`. You just need to set these two values in `speech.py` file in `main` function.

Then just run:

```bash
python3 train_speech_model.py
```

### Code

```python
# Decide for number of gaussian mixture components used for the target model
M_target = 5
M_non_target = 5

# Initialize mean vectors to randomly selected data points from corresponding class
MUs_target = train_target[randint(1, len(train_target), M_target)]
MUs_non_target = train_non_target[randint(1, len(train_non_target), M_non_target)]

# Initialize all variance vectors (diagonals of the full covariance matrices) to
# the same variance vector computed using all the data from the given class
COVs_target = [np.var(train_target, axis=0)] * M_target
COVs_non_target = [np.var(train_non_target, axis=0)] * M_non_target
```

Weights are trained with two different approaches:
- uniform,
- random.

```python
# Use uniform distribution as initial guess for the weights
Ws_target = np.ones(M_target) / M_target
Ws_non_target = np.ones(M_non_target) / M_non_target

# ------------------------------------------------------------
Ws_target = np.rand(M_target)
Ws_non_target = np.rand(M_non_target)
```

#### Score probability

```python
ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)
score = sum(ll_target) + np.log(P_target) - (sum(ll_non_target) + np.log(P_non_target))
```

If the score is above `score_border` value, the model predicts that the tst file is target, otherwise it is not target. All the score border tests were calculated as average from 10 different runs without augmented data.

**Uniform weights distribution:**
| Score border  | Correctness of target | Correctness of non-target | Correctness average |
| ------------  | --------------------: | ------------------------: | ------------------: |
| -200          | **100.00%**           | 87.00%                    | 93.50%              |
| 0             | 99.00%                | 88.83%                    | 93.92%              |
| 200           | **100.00%**           | 90.83%                    | 95.42%              |
| 400           | 97.00%                | 94.17%                    | **95.58%**          |
| 600           | 93.00%                | **97.00%**                | 95.00%              |

**Random weights distribution:**
| Score border  | Correctness of target | Correctness of non-target | Correctness average |
| ------------  | --------------------: | ------------------------: | ------------------: |
| -200          | **100.00%**           | 86.67%                    | 93.33%              |
| 0             | **100.00%**           | 89.33%                    | 94.67%              |
| 200           | 99.00%                | 91.50%                    | 95.25%              |
| 400           | 95.00%                | 94.33%                    | 94.67%              |
| 600           | 96.00%                | **95.33%**                | **95.67%**          |

After 100 iterations of EM agorithm to train the two GMMs from target and non target we have:
```
Iteration: 99  Total log-likelihoods: -301292.9122690845 for target; -2359995.631752151 for non targets.
```

The `TTL_target` value (-301292.9122690845) is significantly higher (less negative) than the `TTL_non_target` value (-2359995.631752151), indicating that the model is much better at explaining the data from the target class compared to the non-target class.

We see that there could be reason to set score border to different number than 0. Model is better recognizing when speech is target, when it is not, the model is little bit prone to guessing that the input is target. Weights distribution is not significant for model correctness.

I also tried to get rid of EM algorithm training, but the probability was worse. Increasing number of iteration in EM algorithm training led to slower training but better results. So I set the number of training iterations to 50.


#### Average probability

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

The average probability is worse than `score prediction`.

#### Max probability

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

This is because in every .wav file there is probability somewhere around 0.99 that the frame is from target file. So this approach is useless for our problem.

## Data augmentation

We implemented data augmentation for training data, the code is provided below:

```python
def augment_audio(wav_data, augmentation_factor):
    """
    Augment audio data by time stretching
    """
    wav_data_float = librosa.util.buf_to_float(wav_data)
    augmented_data = librosa.effects.time_stretch(wav_data_float, rate=augmentation_factor)
    return augmented_data

def augment_add_random_noise(wav_data, noise_level=0.01):
    """
    Add random noise to audio data.

    Parameters:
    - wav_data: NumPy array representing the audio signal.
    - noise_level: Magnitude of the random noise (default is 0.01)

    Returns:
    - Noisy audio signal as a NumPy array.
    """
    # Generate random noise with the same length as the audio signal
    noise = noise_level * np.random.rand(len(wav_data))

    # Add the noise to the audio signal
    noisy_audio = wav_data + noise

    return noisy_audio

def wav16khz2mfcc(dir_name, augment=False):
    """
    Loads all *.wav files from directory dir_name (must be 16KHz), converts them into MFCC
    features (13 coefficients) and stores them into a directory. Keys are the file names
    and values and 2D numpy arrays of MFCC features.
    """
    features = {}
    for f in glob(dir_name + '/*.wav'):
        print('Processing file: ', f)
        rate, s = wavfile.read(f)
        assert(rate == 16000)
        features[f] = mfcc(s, 400, 240, 512, 16000, 23, 13)
        if augment:
            stretch_speeds = [0.5, 0.8, 1.2, 1.5, 2.0]
            for index, speed in enumerate(stretch_speeds):
                augmented_data = augment_audio(s, speed)
                augmented_features = mfcc(augmented_data, 400, 240, 512, 16000, 23, 13)
                augmented_key = f.replace('.wav', '_augmented_stretch_speed_' + str(index) + '.wav')
                features[augmented_key] = augmented_features
            augmented_noisy_data = augment_add_random_noise(s)
            augmented_noisy_features = mfcc(augmented_noisy_data, 400, 240, 512, 16000, 23, 13)
            augmented_noisy_key = f.replace('.wav', '_augmented_random_noise.wav')
            features[augmented_noisy_key] = augmented_noisy_features
    return features
```

At the beginning the results were bad, the correctness of non target data dropped to somewhere around 50-60%. We solved this with increasing `M_target` and `M_non_target` number of gaussian mixture components used for the target and non target models. It had a negative impact on the training time of the model, but the average percentage between target and non target data went back to 90+%.

We added wav files with different stretch speeds, as we can see we used 5 different stretch speeds for augmentation and also we created augmented audio file with noise. So instead of having 20 target data and 132 non target data we have 140 target data and 924 non target data.