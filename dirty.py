#!/usr/bin/env python3

from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import pickle
import sys

from util import encode, preprocess, save, tick

def train(data):
    X = preprocess.field_array(data, 'tfidf')
    X = pad_sequences(X, maxlen=2368, value=0)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = [y-1 for y in preprocess.field_array(data, 'Class')]
    y = np_utils.to_categorical(y, 9)

    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(9, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    ckpt = ModelCheckpoint('best_model_saving_path', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    model.add(BatchNormalization())
    model.fit(X, y, batch_size=1000, epochs=10, validation_split=0.2, callbacks=[ckpt])
    model.save('final_model_saving_path')

if '__main__' == __name__:

    #tr = preprocess.load('src/training_variants', 'src/training_text') # {tr,te,tte}.pkl, 0:00:01.341790
    #pickle.dump(tr, open('tmp/tr.pkl', 'wb'))

    #tr = pickle.load(open('tmp/tr.pkl', 'rb'))
    #preprocess.remove_stop_words(tr) # *.rsw.pkl, 0:00:44.942251
    #pickle.dump(tr, open('tmp/tr.rsw.pkl', 'wb'))

    #tr = pickle.load(open('tmp/te.rsw.pkl', 'rb'))
    #preprocess.normalize_target_variation(tr) # 0:00:11.831447
    #preprocess.replace_text(tr, in_field='Variation', to_str=' __TARGET_VARIATION__ ') # 0:00:00.419632
    #preprocess.sentences(tr) # *.s.pkl, 0:01:15.682816
    #pickle.dump(tr, open('tmp/te.s.pkl', 'wb'))

    #tr = pickle.load(open('tmp/tr.s.pkl', 'rb'))
    #preprocess.paragraph_by_variation(tr, 0) # *.pbvw*.pkl, 0:00:00.929176
    #pickle.dump(tr, open('tmp/tr.pbvw0.pkl', 'wb'))

    #tr = pickle.load(open('tmp/tr.pbvw0.pkl', 'rb'))
    #tsm = encode.tfidf_sequential_model(tr) # tr.tsm.pkl, 0:00:04.937104
    #pickle.dump(tsm, open('tmp/tr.tsm.pkl', 'wb'))

    #tr = pickle.load(open('tmp/tr.pbvw0.pkl', 'rb'))
    #c = preprocess.concatenate(tr) # 0:00:00.530155
    #tsm = encode.tfidf_sequential_model(c, False) # tr.c.tsm.pkl, 0:00:02.323202
    #pickle.dump(tsm, open('tmp/tr.c.tsm.pkl', 'wb'))

    #tsm = pickle.load(open('tmp/tr.tsm.pkl', 'rb')) # use tr.{tsm,c.tsm}.pkl
    #tr = pickle.load(open('tmp/tr.pbvw0.pkl', 'rb'))
    #encode.tfidf_sequential(tr, tsm) # tr.ts.pkl, 0:08:12.010365
    #save(tr, 'tmp/tr.ts.pkl')
    #tr = pickle.load(open('tmp/te.pbvw0.pkl', 'rb'))
    #encode.tfidf_sequential(tr, tsm)
    #save(tr, 'tmp/te.ts.pkl')

    #tr = pickle.load(open('tmp/tr.ts.pkl', 'rb'))
    #te = pickle.load(open('tmp/te.ts.pkl', 'rb'))
    #train(tr)

#   preprocess.normalize_gene(tr) # not yet
#   preprocess.replace_text(tr, in_field='Gene', to_str=' __TARGET_GENE__ ')
#   preprocess.replace_classified_variant(tr) # not yet
#   tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#   tf['svd'] = encoding.svd(tr['tfidf'], 50)
#   tf['d2v'] = encoding.doc2vec(tr, 50)
#   tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4
