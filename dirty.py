#!/usr/bin/env python3

opts = {
    'batch_size': 0.8,
    'embedding_size': 8,
    'epochs': 100000,
    'lr': 1e-3,
    'min_delta': 1e-4,
    'patience': 100,
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
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import util
from util import encode, preprocess

def cnn_train(data):
    X = util.field_array(data, 'X')
    X = pad_sequences(X, maxlen=2000, value=0)
    #X = X.reshape(X.shape[0], X.shape[1], 1)
    y = [y-1 for y in util.field_array(data, 'y')]
    y = np_utils.to_categorical(y, 9)

    model = Sequential()
    model.add(Embedding(np.amax(X)+1, opts['embedding_size'], input_length=2000))
    #model.add(Conv1D(64, 2, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Conv1D(16, 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Conv1D(16, 4, activation='relu'))
    model.add(MaxPooling1D(4))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    #model.add(Conv1D(16, 2, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    #model.add(Conv1D(16, 2, activation='relu'))
    #model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
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

def rnn_train(data):
    X = util.field_array(data, 'X')
    X = pad_sequences(X, maxlen=2000, value=0)
    #X = X.reshape(X.shape[0], X.shape[1], 1)
    y = [y-1 for y in util.field_array(data, 'y')]
    y = np_utils.to_categorical(y, 9)

    model = Sequential()
    model.add(Embedding(np.amax(X)+1, opts['embedding_size'], input_length=2000))
    #for _ in range(3):
    #    model.add(LSTM(opts['output_dim'], return_sequences=True))
    model.add(LSTM(opts['output_dim']))
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

    #tr = preprocess.load('src/training_variants', 'src/training_text') # {tr,te,tte}.pkl, 0:00:01.341790
    #save(tr, 'tmp/tr.pkl')

    #tr = load('tmp/tr.pkl')
    #preprocess.remove_stop_words(tr) # *.rsw.pkl, 0:00:44.942251
    #save(tr, 'tmp/tr.rsw.pkl')

    #tr = util.load('tmp/tr.rsw.pkl')
    #preprocess.normalize_target_variation(tr) # 0:00:11.831447
    #preprocess.replace_text(tr, in_field='Variation', to_str=' __TARGET_VARIATION__ ') # 0:00:00.419632
    #preprocess.sentences(tr) # *.s.pkl, 0:01:15.682816
    #util.save(tr, 'tmp/tr.s.pkl')

    #tr = util.load('tmp/tr.s.pkl')
    #preprocess.paragraph_by_variation(tr, 0) # *.pbvw*.pkl, 0:00:00.929176
    #util.save(tr, 'tmp/tr.pbvw0.pkl')

    #tr = util.load('tmp/tr.pbvw0.pkl')
    #c = preprocess.concatenate(tr) # 0:00:00.530155
    #tsm = encode.tfidf_sequential_model(tr) # 0:00:04.937104
    #tsm = encode.tfidf_sequential_model(c, False) # 0:00:02.323202

    #encode.tfidf_sequential(tr, tsm) # *(.c).ts.pkl, 0:08:12.010365
    #util.save(tr, 'tmp/tr.ts.pkl', 'tfidf', 'Class')

    #encode.dummy_sequential(tr, tsm) # *.ds.pkl, 0:03:46.821351
    #util.save(tr, 'tmp/te.ds.pkl', 'dummy', 'Class')

    #tr = load('tmp/tr.ts.pkl')
    #encode.sparse_clean(tr, 'tfidf', 0.05) # *(.c).ts.sc*.pkl, 0:07:57.079560
    #save(tr, 'tmp/tr.ts.sc005.pkl', 'tfidf', 'Class')

    #tr = util.load('tmp/te.pbvw0.pkl')
    #tte = util.load('src/trueTstTotal.pkl')
    #tr = preprocess.subset(tr, tte) # tte(.c).ts.pkl, 0:00:00.180169
    #encode.tfidf_sequential(tr, tsm)
    #encode.dummy_sequential(tr, tsm)
    #util.save(tr, 'tmp/tte.ds.pkl', 'dummy', 'Class')

    tr = pickle.load(open('tr.ds.pkl', 'rb'))
    #cnn_train(tr)
    rnn_train(tr)

#   preprocess.normalize_gene(tr) # not yet
#   preprocess.replace_text(tr, in_field='Gene', to_str=' __TARGET_GENE__ ')
#   preprocess.replace_classified_variant(tr) # not yet
#   tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#   tf['svd'] = encoding.svd(tr['tfidf'], 50)
#   tf['d2v'] = encoding.doc2vec(tr, 50)
#   tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4
