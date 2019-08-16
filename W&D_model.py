import numpy as np
import pandas as pd


from keras.models import Model
from keras.layers import Dense, concatenate, Input
from keras.optimizers import Adam, SGD
from keras.layers import Flatten, concatenate, Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2, l1_l2

from tensorflow.python.keras.datasets import imdb

word_num = 6666

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=word_num)


def vectorize_sequences(sequences, dimension=word_num):
    results = np.zeros((len(sequences), dimension))  # 数据集长度，每个评论维度10000
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # one-hot
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')  # 向量化标签数据
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:2000]
partial_x_train = x_train[2000:]

y_val = y_train[:2000]
partial_y_train = y_train[2000:]


wide_input = Input(shape=(word_num,))
wide = Dense(100, activation='relu')(wide_input)

deep_input = Input(shape=(word_num,))
deep = Dense(100, activation='relu')(deep_input)
deep = Dense(100, activation='relu')(deep)
deep = Dense(50, activation='relu')(deep)
deep = Dense(20, activation='relu')(deep)

wd_input = concatenate([wide, deep])
wd = Dense(1, activation='sigmoid')(wd_input)

model = Model(inputs=[wide_input, deep_input], outputs=wd)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])
model.fit([partial_x_train, partial_x_train], partial_y_train, epochs=20, batch_size=200)


deep = Dense(1, activation='sigmoid')(deep)
deep_model = Model(inputs=deep_input, outputs=deep)
deep_model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])
# deep_model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=200)