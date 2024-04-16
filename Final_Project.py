#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:09:52 2024

@author: oumousamake
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_images_and_labels(data_dir, selected_letters, size):
    images = []
    labels = []

    for letter_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, letter_dir)) and letter_dir in selected_letters:
            for file in os.listdir(os.path.join(data_dir, letter_dir)):
                if file.endswith('.jpg'):
                    image_path = os.path.join(data_dir, letter_dir, file)
                    image = cv2.imread(image_path)
                    image_resized = cv2.resize(image, (size, size))
                    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                    image_normalized = image_gray / 255.0 
                    label = file[0]
                    images.append(image_normalized)
                    labels.append(label)

    return np.array(images), np.array(labels)

data_dir = r'/Users/oumousamake/Downloads/archive (1)/asl_alphabet_train/asl_alphabet_train'
selected_letters = ['A', 'B', 'F', 'I', 'J', 'L', 'O', 'P', 'V', 'Y']
size = 100

# Load training data
images, labels = load_images_and_labels(data_dir, selected_letters, size)

# Load training data
images, labels = load_images_and_labels(data_dir, selected_letters, size)

# Flatten images
images_flat = images.reshape(images.shape[0], -1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

# Define and train PCA for dimensionality reduction (optional)
pca = PCA(n_components=100)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define and train the decision tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_pca, y_train)

# Evaluate the classifier
train_accuracy = accuracy_score(y_train, dt_classifier.predict(X_train_pca))
test_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test_pca))

# Evaluate the classifier
train_accuracy = accuracy_score(y_train, dt_classifier.predict(X_train_pca))
test_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test_pca))

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, dt_classifier.predict(X_test_pca), target_names=selected_letters))


# Define and train the decision tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=None, class_names=selected_letters)
plt.show()
