# optical-character-recognition
A simple OCR model for 26 alphabetic and 10 numeric characcters

Dataset used - the 'older version' dataset from the source https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset

# How to run?
1. Download the dataset and unzip it. The unzipped data should be by default be in a directory named 'archive'
2. The pre-processing.ipynb notebook creates all the required sub-direcotries using the OS module. Directly run it to finish the pre-processing
3. Run the training-model.ipynb notebook to train the model and save it if neccessary with the last line thats commented out

# Requirements
pytorch, 
numpy, 
matplotlib, 
CUDA version compatible with specific GPU

# Results
Accuracy acheived - 84.72%
