#!/usr/bin/python
from keras.models import Sequential, Model
from keras.layers import LSTM, Embedding, Dense
from keras.optimizers import Adam

def _lstm_model(input_dim, input_length):
    output_dim = 16
    optimizer = Adam(lr = 0.05)

    model = Sequential()
    model.add(LSTM(output_dim, input_shape = (input_length, input_dim), return_sequences = True))

    for _ in range(3):
        model.add(LSTM(output_dim, return_sequences = True))

    model.add(LSTM(output_dim))
    model.add(Dense(9, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['categorical_crossentropy'])
    return model


def _tokenizer_model(input_dim, input_length):
    output_dim = 32
    optimizer = Adam(lr = 0.033)

    model = Sequential()
    model.add(Embedding(input_dim, output_dim, input_length = input_length))

    for _ in range(3):
        model.add(LSTM(output_dim, return_sequences = True))

    model.add(LSTM(output_dim))
    model.add(Dense(9, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['categorical_crossentropy'])
    return model
