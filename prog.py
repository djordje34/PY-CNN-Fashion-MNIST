import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
TEST_SIZE = 0.2
RANDOM_STATE = 123
NO_EPOCHS = 10
BATCH_SIZE = 128

def getData(train=1):
    if train:
        return pd.read_csv('fashion-mnist_train.csv')
    
    return pd.read_csv('fashion-mnist_test.csv')


def dataPreprocessing(data):
    out_y = tf.keras.utils.to_categorical(data.label, NUM_CLASSES)
    #similar to onehot->convert labels to arrs 
    print(out_y)
    num_images = data.shape[0]
    x = data.values[:,1:]
    x_shaped = x.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    out_x = x_shaped / 255
    return out_x, out_y

def getModel():#simple seq model, cnn za slike
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    kernel_initializer='he_normal',
                    input_shape=(IMG_ROWS, IMG_COLS, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 
                    kernel_size=(3, 3), 
                    activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(NUM_CLASSES, activation='softmax'))


    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
    
    
    return model

def main():
    train,test = getData(),getData(0)
    
    trX,trY = dataPreprocessing(train)#convert 1d lista u wXh nested list
    teX,teY = dataPreprocessing(test)

    X_train, X_val, y_train, y_val = train_test_split(trX, trY, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    #passovan rand int -> rezultati reproducible
    model = getModel()
    train_model = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val))


    score = model.evaluate(teX, teY, verbose=0)
    print('Acc:', score[1])

if __name__=='__main__':
    main()