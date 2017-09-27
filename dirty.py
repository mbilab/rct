#!/usr/bin/env python3

opts = {
    'batch_size': 0.8,
    'embedding_size': 32,
    'epochs': 100000,
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
import pickle
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Embedding, Flatten, GlobalAveragePooling1D, LSTM, MaxPooling1D
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import util
try:
    from util import encode, preprocess
except:
    pass

def cnn_train(data):
    X = util.field_array(data, 'X')
    X = pad_sequences(X, maxlen=opts['input_length'], value=0)
    #X = X.reshape(X.shape[0], X.shape[1], 1)
    y = [y-1 for y in util.field_array(data, 'y')]
    y = np_utils.to_categorical(y, 9)

    model = Sequential()
    model.add(Embedding(np.amax(X)+1, opts['embedding_size'], input_length=opts['input_length']))
    #model.add(Conv1D(64, 2, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Conv1D(32, 8, activation='relu', padding='same'))
    model.add(Conv1D(64, 2, activation='relu', padding='same'))
    model.add(MaxPooling1D(opts['input_length']))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    #model.add(BatchNormalization())
    #model.add(Dropout(0.8))
    model.add(Dense(9, activation='softmax'))

    optimizer = Adam(lr=opts['lr'])
    model.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'], optimizer=optimizer)
    #model.summary()
    #return

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

if '__main__' == __name__:

    #tr = preprocess.load('src/te.csv') # tr.pkl, 0:00:01.650411
    #util.save(tr, 'tmp/te.pkl')

    #tr = util.load('tmp/te.pkl')
    #preprocess.remove_stop_words(tr) # tr.rsw.pkl, 0:00:46.640691
    #util.save(tr, 'tmp/te.rsw.pkl')

    #tr = util.load('tmp/tr.rsw.pkl')
    #preprocess.normalize_target_variation(tr) # 0:00:11.750795
    #preprocess.replace_text(tr, in_field='Variation', to_str=' __TARGET_VARIATION__ ') # 0:00:00.348791
    #preprocess.sentences(tr) # *.s.pkl, 0:01:16.074815
    #util.save(tr, 'tmp/tr.s.pkl')

    #util.swv('tmp/te.rsw.pkl') # 0:01:10.617185

    #tr = util.s2ds('tmp1/tr.s.pkl', 0, True, 1e-4, 'tmp1/tte.s.pkl') # 0:07:59.554650

    #encode.tfidf_sequential(tr, tsm) # *(.c).ts.pkl, 0:08:12.010365
    #util.save(tr, 'tmp/tr.ts.pkl', 'tfidf', 'Class')

    #tr = util.load('tmp/tmp2.pkl')
    #encode.sparse_clean(tr, 0.5) # *(.c).ts.sc*.pkl, 0:07:57.079560
    #util.save(tr, 'tmp/tmp.pkl')

    #tr = util.load('ignore/stage_1_tmp/te.pkl')
    #tte = util.load('src/stage_1/trueTstTotal.pkl')
    #tr = preprocess.subset(tr, tte) # tte(.c).ts.pkl, 0:00:00.180169
    #encode.tfidf_sequential(tr, tsm)
    #encode.dummy_sequential(tr, tsm)
    #util.save(tr, 'tmp1/tte.pkl')

    #tr = util.load('tmp/tr.dsc-4.pkl')
    #util.histogram(tr)

    #tr = pickle.load(open('tr.ds.pkl', 'rb'))
    #cnn_train(tr)
    #rnn_train(tr)

    #predict('tmp1/tr.dsc-4.h5', 'tmp1/tte.dsc-4.pkl', 'tmp1/tte.pkl')

#   preprocess.normalize_gene(tr) # not yet
#   preprocess.replace_text(tr, in_field='Gene', to_str=' __TARGET_GENE__ ')
#   preprocess.replace_classified_variant(tr) # not yet
#   tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#   tf['svd'] = encoding.svd(tr['tfidf'], 50)
#   tf['d2v'] = encoding.doc2vec(tr, 50)
#   tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4
