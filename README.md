# Kaptioner ML
This Repostitory contains the files that form the Machine Learning Model for the Kaptioner Web App https://github.com/PrinceGupta1999/Kaptioner .
we have built a Image to Sequence Model consisting of an Encoder which makes use of Convolutional Nueral Networks and processes all the images and generates a corresponding feature vector, A Decoder which processes the feature vector using Recurrent Nueral Network consisting of LSTM layers and gives the desired output Information Labels.

We have taken a Pre-Trained ResNet Model, then we have freezed the last few layers and train it.

## Contents

### Preprocessing
* [build_vocab.py](https://github.com/ishankjain/Kaptioner-ML/build_vocab.py) : Creates a vocabulary of all the Identification Labels in the given dataset
* [resize.py](https://github.com/ishankjain/Kaptioner-ML/resize.py) : Resizes all the images to a given size (224 * 224) and then analyze and load the data

### Training and Testing
* [train.py](https://github.com/ishankjain/Kaptioner-ML/train.py) : Analyzes the images in the dataset in then trains the Model 
* [sample_test.py](https://github.com/ishankjain/Kaptioner-ML/sample_test.py) : Accepts an image path as parameter and returns the predicted Caption as Output
