import numpy as np

from keras.layers import Dense, Input, concatenate
from keras.optimizers import SGD
from keras.models import Model

from tensorflow.python.keras.datasets import imdb

word_num = 6666

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=word_num)


def vectorize_sequences(sequences, dimension=word_num):  # one-hot
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 
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
