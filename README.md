# sur

[assignment](https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2023-2024/SUR_projekt2023-2024.txt)

### IMG

1. Go to directory
```
cd src/img
```
2. Create virtual env 
```
python3 -m venv .venv
```
3. Activate virtual env 

Unix:
```
source .venv/bin/activate
```
Windows:
```
.venv\Scripts\activate
```
4. Install requirements
```
python3 -m pip install -r requirements.txt
```
5. Run
```
python3 image_NN.py
```
- If `one_person_detector.keras` file is available model will be loaded from this file, otherwise it will be trained. For training data in `../data` will be used.

- For evaluation data script uses data with path specified in variable ``PATH_EVAL``.

- Script writes the output of evaluation to the `../image_NN.txt` file

##### TODO: Img solution summary

### AUDIO

### COMBINED
