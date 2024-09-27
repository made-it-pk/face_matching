# Face Matching

## 1. Overview
This repo offers code for training,testing, and inference for Siamese Neural Networks for face recognition. The code can also be adapted for other use cases beyond face recognition.

## 2. Features
- Switch network backbone
- Use wandb for logging training experiments
- Test/evaluate
- Run inference
- Use KNN to improve detection results

## 3. Setup
1. Create a python 3.8+ environment using miniconda:
~~~bash
conda create -n face_matching python=3.8
~~~
2. Activate the environment.
3. Run:
~~~bash
pip install -r requirements.txt
~~~

## 4. Train
1. Adjust config in config.py
2. Use date_preprocessing.py to generate triplets.
3. Use main.py to train.

## 5. Test/Evaluate
1. Use testing_main.py to test.

## 6. Inference
1. We provide jupyter notebook showing example inference code at `facenet_knn/main.ipynb`

## TODO
- [ ] Refactoring
