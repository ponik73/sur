## SPEECH

This speech documentation describes the approach that we used for deciding if the .wav file was recorded by the target person or not. It outlines methods that we use for training and how we evaluate the test data or any unseen data.

### How to run the program

1. Go to the directory `speech`.
2. Install dependencies using 

```bash
python3 -m pip install -r requirements.txt
```

3. You have two options:
   1. You can run training
   2. You can run an evaluation of the test set

Running training:

```bash
python3 train_speech_model.py
```

Running evaluation:

```bash
python3 evaluate.py
```

### Train speech model code

1. Loading files and data augmentation
   We used the `wav16khz2mfcc` function from the provided codes and added data augmentation for different stretch speeds, we also augmented the file with random noise. With these two techniques, we created six new augmented data (five with stretching speed and one with adding noise). The function is in the `utilities.py` file. Data augmentation is also described [below](#data-augmentation). Also we implemented trimming silence from both train and dev speeches using `librosa` library.
2. Decide the number of Gaussian mixture components used for the target and nontarget models.
   We tried to change the number of Gaussian mixture components used for the target and non-target models, firstly we started with 5 components for both target and non-target models. Without data augmentation, the model worked fine but after augmentation was added, the correctness of the model rapidly decreased. We then tried to set the numbers to 25 which led to prolonged training but improved the model correctness back to the 90%+ as was before with 5 components and without data augmentation. We then decreased the number of components to 15 to avoid slow training and measured model correctness which was again above 90%.
3. Initialize all variance vectors.
   We kept this code from the provided codes.
4. Generating weights
   Here we tried two different approaches as initial guesses for the weights. Firstly we used uniform distribution and then we tried random weights distribution. The distinct distributions were examined on non-augmented data.

   ```python
    # Use uniform distribution as initial guess for the weights
    Ws_target = np.ones(M_target) / M_target
    Ws_non_target = np.ones(M_non_target) / M_non_target

    # Use random distribution as initial guess for the weights
    Ws_target = np.rand(M_target)
    Ws_non_target = np.rand(M_non_target)
    ```

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

    If the score is above `Score border` value, the model predicts that the tst file is target, otherwise it is not target. All the score border tests were calculated as average from 10 different runs. Weight distribution is not significant for model correctness.
5. We are running n iterations to get distinct results.
   We set the number of iterations for each training before starting. The program runs n iterations and calculates the correctness of the target model, the correctness of the non-target model, and also the average correctness which is calculated as `(correctness_target + correctness_non_target) / 2`  and this average correctness gives us a better understanding of how good the parameters are. For each iteration:
   1. Initialize mean vectors to randomly selected data points from corresponding class. This code is from the provided codes.
   2. We measure the time for training the two GMMs for target and non-target models. In this step, we tried different numbers of iterations, with a maximum of 100. This training was slow and took 650 seconds on Windows which is almost 11 minutes. From the log information about total log-likelihoods, we discovered that the convergence is mostly around the 30th iteration which is the same as in the provided codes. 30 iterations took approximatelly 198 seconds on Windows. We also tried to eliminate EM algorithm training, but the correctness decreased. Increasing the number of iterations in EM algorithm training led to slower training but better results. After convergence, the results also converged.

After 90 iterations of the EM algorithm to train the two GMMs from target and non-target, we have had:
```
Iteration: 90  Total log-likelihoods: -2314243.0034861485 for target; -17922917.091317676 for non targets.
```

The `TTL_target` value (-2314243.0034861485) is significantly higher (less negative) than the `TTL_non_target` value (-17922917.091317676), indicating that the model is much better at explaining the data from the target class compared to the non-target class.

We see that there could be a reason to set the score border to a different number than 0. The model is better at recognizing when speech is the target, when it is not, it is a little prone to guessing that the input is the target.

   3. After training parameters, we evaluate test data for both target and non-target data. The results we save to `score_CLASS` lists with scores for each data. Then we try to apply different borders to discover which can split the data most accurately. We log information about the target model correctness, the non-target model correctness, and the average correctness.
   4. We also remember the highest `max_avg_correctness` and if some border and parameters outperform the current maximum, we save these parameters into `.txt` files for evaluation. We also save that `border` and `max_avg_correctness` data into files, border for evaluation and maximum correctness for the next training.

Below we describe three main approaches that we tried to use to decide if the tested data is target or not. The best was the `Score` approach closely followed by the `Average probability` approach. The `Max probability` approach was not working. In the [Score](#score) section we also summarize the results of the best run.

#### Score

```python
ll_target = logpdf_gmm(tst, Ws_target, MUs_target, COVs_target)
ll_non_target = logpdf_gmm(tst, Ws_non_target, MUs_non_target, COVs_non_target)
score = sum(ll_target) + np.log(P_target) - (sum(ll_non_target) + np.log(P_non_target))
```

Correctness is described in the weights distribution tables above (The scoring approach was used there). Our best found is described in the table below.

| Score border | EM algorithm training iterations | Gausian mixture components | Average correctness |
| :----------: | :------------------------------: | :------------------------: | :-----------------: |
| -200         | 30                               | 15 for both                | 99.17%              |

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

### Data augmentation

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

In the beginning, the results were bad, the correctness of non-target data dropped to somewhere around 50-60%. We solved this by increasing the `M_target` and `M_non_target` Gaussian mixture components used for the target and non-target models. It had a negative impact on the training time of the model, but the average correctness percentage between the target and non-target data went back to 90%+.

We added new augmented data with different stretch speeds. As we can see, we used 5 different stretch speeds for augmentation and we also created an augmented audio file with noise. So instead of having 20 target data and 132 non-target data, we have 140 target data and 924 non-target data.

### Evaluate data

In the main function, there is a `wav16khz2mfcc` function that takes the directory path as the parameter. It evaluates all the WAV files in that directory and prints results in the desired format.

All the parameters are loaded from TXT files, which were created by training speech model.
