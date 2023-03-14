# PY-CNN-Fashion-MNIST
Convolutional NN for classification of images

## About Dataset

Link leading to dataset [GitHub Pages](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

Dataset contains 60000 images for training, which is split into 20% for validation and then rest for training (sklearn train_test_split with random-state=123).

File test.csv contains 10000 images with proper labeling for testing purposes.


## Network

_________________________________________________________________
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param num.   
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
dropout_1 (Dropout)          (None, 11, 11, 64)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                16010     
_________________________________________________________________
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________

For given batch size=128 and number of epochs=10:

Accuracy on the training set: 90.51%

Accuracy on the test set: 91.22%

## Dependencies

1. tensorflow
2. pandas
3. sklearn (sklearn.model_selection)
