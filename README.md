# ASL Image Classifier

## Program Description
This program implements a machine learning-based classifier to recognize American Sign Language (ASL) alphabet symbols represented in images of hand gestures. The classifier is trained on a dataset consisting of images of hands performing ASL symbols, with each image labeled with the corresponding letter it represents. This code only classifies a subset of the alphabet, consisting of 10 letters: A, B, F, I, J, L, O, P, V, and Y.

## Instructions
* Download [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* Unzip in the same directory the files with algorithms are
* Make sure all modules are installed in your computer!

## The algorithm follows the following steps:
### Data Preprocessing:
* Read images from the dataset directory
* Resize images to a standardized size
* Convert images to grayscale
* Normalize pixel values

### Feature Extraction:
* Apply Principal Component Analysis (PCA) for dimensionality reduction

### Model Training:
* Train a Support Vector Machine (SVM) classifier on the reduced feature space obtained from PCA
  

