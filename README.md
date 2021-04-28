# CRNNForFer2013

## Environment
python == 2.7.18
pytorch == 1.3.1
scikit-learn == 0.20.4
matplotlib == 1.5.1

## Introduction
### This is a C-RNN model for fer2013 expression recognition data set, which is composed of VGG19 and LSTM.
### We preprocess the data set by data balance operation (data undersampling and data oversampling), and save the processed files in the “data” folder.
### Run preprocess.py to convert the images and labels to. H5 format.
### Run train.py to train the model.
### Run predict to evaluate the model.
