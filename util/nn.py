#!/usr/bin/env python3

opts = {
    'batch_size': 0.05,
    'embedding_size': 256,
    'epochs': 20,
    'lr': 1e-3,
    'min_delta': 1e-4,
    'input_length': 1000,
    'patience': 10,
    'random_state': 0,
    'regularizer': 0,
    'validation_split': 0.2,
}

import numpy as np
np.random.seed(opts['random_state'])

from inspect import signature
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Embedding, Flatten, GlobalAveragePooling1D, Input, LSTM, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import util
from random import shuffle

def cnn_train(data):
    X, y, X_val, y_val = format_data(data)

    model = Sequential()
    model.add(Embedding(np.amax(X)+1, opts['embedding_size'], input_length=opts['input_length']))
    model.add(Conv1D(32, 4, activation='relu', padding='same'))
    model.add(Conv1D(64, 2, activation='relu', padding='same'))
    model.add(MaxPooling1D(opts['input_length']))
    model.add(Dropout(0.5))
    model.add(Flatten())
    #model.add(BatchNormalization())
    #model.add(Dropout(0.8))
    model.add(Dense(9, activation='softmax'))

    compile_and_fit(model, X, y, X_val, y_val)
    return X_val, y_val

def cnn2_train(data):
    X, y, X_val, y_val = format_data(data)

    inputs = Input(shape=(opts['input_length'],))
    common = Embedding(np.amax(X)+1, opts['embedding_size'])(inputs)
    models = []

    model = Conv1D(32, 4, activation='relu', padding='same')(common)
    model = Conv1D(64, 2, activation='relu', padding='same')(model)
    model = MaxPooling1D(opts['input_length'])(model)
    model = Dropout(0.5)(model)
    model = Flatten()(model)
    models.append(model)

    model = Conv1D(32, 6, activation='relu', padding='same')(common)
    model = Conv1D(64, 2, activation='relu', padding='same')(model)
    model = MaxPooling1D(opts['input_length'])(model)
    model = Dropout(0.5)(model)
    model = Flatten()(model)
    models.append(model)

    model = Conv1D(32, 8, activation='relu', padding='same')(common)
    model = Conv1D(64, 2, activation='relu', padding='same')(model)
    model = MaxPooling1D(opts['input_length'])(model)
    model = Dropout(0.5)(model)
    model = Flatten()(model)
    models.append(model)

    model = concatenate(models)
    outputs = Dense(9, activation='softmax')(model)

    model = Model(inputs=inputs, outputs=outputs)

    compile_and_fit(model, X, y, X_val, y_val)
    return X_val, y_val

def compile_and_fit(model, X, y, X_val, y_val):
    optimizer = Adam(lr=opts['lr'])
    model.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'], optimizer=optimizer)
    model.summary()
    # return

    ckpt = ModelCheckpoint('stage_1_parallel_cnn', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(min_delta=opts['min_delta'], patience=opts['patience'])

    if opts['batch_size'] <= 1:
        opts['batch_size'] = int(len(y) * opts['batch_size'])
    keys = set(opts) & set(signature(model.fit).parameters)
    fit_opts = { key: opts[key] for key in keys }
    fit_opts['validation_split'] = 0
    fit_opts['validation_data'] = [X_val, y_val]
    if sys.stdout.isatty():
        verbose = 2
    else:
        verbose = 0
    model.fit(X, y, callbacks=[ckpt, es], verbose=verbose, **fit_opts)
    model.save('final_model_saving_path')

def format_data(data):
    shuffle(data)
    X = util.field_array(data, 'X')
    X = pad_sequences(X, maxlen=opts['input_length'], value=0)
    y = [y-1 for y in util.field_array(data, 'y')]
    y = np_utils.to_categorical(y, 9)
    return X[int(opts['validation_split'] * len(X)) + 1:], \
        y[int(opts['validation_split'] * len(y)) + 1:], \
        X[0:int(opts['validation_split'] * len(X))], \
        y[0:int(opts['validation_split'] * len(y))] \

def predict(model_filename, data_filename, data_info_filename):
    model = load_model(model_filename)
    data = util.load(data_filename)
    data_info = util.load(data_info_filename)
    X = util.field_array(data, 'X')
    X = pad_sequences(X, maxlen=model.get_layer(index=0).input_shape[1], value=0)

    layer_model = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
    h = layer_model.predict(X)
    o = model.predict(X)

    print(','.join(
        ['Gene', 'Variation'] +
        ['H'+str(i+1) for i in range(h.shape[1])] +
        ['O'+str(i+1) for i in range(o.shape[1])]))
    for i in range(len(data)):
        d = data_info[i]
        print(','.join([str(v) for v in [
            d['Gene'], d['Variation']] +
            list(h[i]) +
            list(o[i])]))

def rnn_train(data):
    #X = [[t.tocoo().data for t in d['X']] for d in data]
    X = util.field_array(data, 'X')
    X = pad_sequences(X, maxlen=opts['input_length'], value=0)
    #X = X.reshape(X.shape[0], X.shape[1], 1)
    y = [y-1 for y in util.field_array(data, 'y')]
    y = np_utils.to_categorical(y, 9)

    model = Sequential()
    model.add(Embedding(np.amax(X)+1, opts['embedding_size'], input_length=opts['input_length']))
    #model.add(LSTM(16, input_shape=(100, 10), return_sequences=True))
    #for _ in range(1):
    #    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(9, activation='softmax'))

    optimizer = Adam(lr=opts['lr'])
    model.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'], optimizer=optimizer)

    ckpt = ModelCheckpoint('best_model_saving_path', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(min_delta=opts['min_delta'], patience=opts['patience'])

    if opts['batch_size'] <= 1:
        opts['batch_size'] = int(len(y) * opts['batch_size'])
    keys = set(opts) & set(signature(model.fit).parameters)
    fit_opts = { key: opts[key] for key in keys }
    if sys.stdout.isatty():
        verbose = 2
    else:
        verbose = 0
    print(opts, fit_opts)
    model.fit(X, y, callbacks=[ckpt, es], verbose=verbose, **fit_opts)
    model.save('final_model_saving_path')

# vi:et:sw=4:ts=4
