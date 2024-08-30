
# Fake News Classifier


Fake News Classifier
This project is a machine learning model designed to classify news as either fake or real. The model is built using TensorFlow and Keras, and it leverages LSTM (Long Short-Term Memory) networks for text classification.



## Table of contents

- Project Overview
- Data Preprocessing
- Model Architecture
- Training
- Evaluation
- Dependencies
- Results
- Conclusion

## Project Overview

The goal of this project is to create a model that can effectively differentiate between fake and real news articles based on their titles. The model uses a combination of Natural Language Processing (NLP) techniques and deep learning methods to achieve high accuracy.

## Data Preprocessing
- Pandas: Used for handling and preprocessing the data.
- NLTK: Used for stemming and removing stopwords.
    - The titles of the news articles were cleaned by removing non-alphabetic characters and converting the text to lowercase.
  - Stemming was applied to reduce words to their root forms.
  - Stopwords were removed to reduce noise in the data.
- One-Hot Encoding: The preprocessed text was converted into one-hot encoded vectors.
- Padding: Sequences were padded to ensure uniform input length for the LSTM model.

## Model Architecture
The model was built using TensorFlow and Keras:
- Embedding Layer: Converts input sequences into dense vectors of fixed size.
- LSTM Layer: Captures the sequential dependencies in the data.
- Dense Layer: A single neuron with a sigmoid activation function for binary classification.

## Training
- Train-Test Split: The dataset was split into training and testing sets with a 67-33 ratio.
- Optimizer: Adam optimizer was used with a binary cross-entropy loss function.
- Batch Size: 64
- Epochs: 10

## Evaluation
Confusion Matrix:   array([[3116,  303],
       [ 227, 2389]], dtype=int64)
Accuracy: The model achieved an accuracy of 91.21%.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NLTK
- Pandas
- Scikit-learn

## Results
The model shows strong performance in classifying fake news with an accuracy of 91.21%. The confusion matrix indicates a good balance between precision and recall.

## Conclusion
This project demonstrates the effectiveness of LSTM networks in text classification tasks. With proper preprocessing and model tuning, the classifier can be a valuable tool in detecting fake news.