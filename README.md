# Creating music pieces using Attention Layers and LSTMs

## Overview
This project contains 3 LSTM models which are used to create music pieces after training on a dataset containing Mozarts symphonies

## Requirements 
1. Keras
2. Glob
3. Music21

## Run
1. To run the attention layer models are in .py format in the models directory, please run the preprocessing.py in pre folder before using these models.
2. There is a .ipynb file for the LSTM model without attention layer in the model directory

## Train
To train from scratch please provide midi files of the music in a folder named data. The final structure must look like this data\*.midi. After this you run preprocessing.py for the attention models and then the model.py in model folder.
For the normal LSTM model, after making the dataset in the desired format, run the cells one by one.

Custom weights on this dataset are available at https://www.dropbox.com/s/v5wr1kb2y1gvyfq/weights%281%29.hdf5?dl=0

## Output
We have provided the output file of what the music will sound like as test_output.mp4

## Loss
Final loss 0.21

## Training time
16 hours using tesla K40 gpu
