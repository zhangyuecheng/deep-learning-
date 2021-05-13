from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM,Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import optimizers
from tensorflow.keras.layers import TimeDistributed
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


from sklearn.preprocessing import LabelEncoder


def create_model_CNN(time_step,vocabulary_size,embedding_size,embedding_matrix,nb_filter,kernel_size):
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=time_step,
                        weights=[embedding_matrix], trainable=False))

    model.add(Conv1D(filters=nb_filter,
                     kernel_size=kernel_size,
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model
