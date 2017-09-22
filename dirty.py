#!/usr/bin/env python3

from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.utils import np_utils
import pickle
import sys

from util import encode, preprocess

def train(data):
    X = preprocess.field_array(data, 'tfidf')
    y = [y-1 for y in preprocess.field_array(data, 'Class')]
    y = np_utils.to_categorical(y, 9)

    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=(train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    ckpt = ModelCheckpoint('best_model_saving_path', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
    tb= TensorBoard(log_dir = 'log_saving_dir' + model_stat, histogram_freq=0, write_graph=True, write_images=True)

    model.fit(train, y, batch_size=10, epochs=1000, validation_split = 0.2, callbacks = [tb, ckpt])
    model.save('final_model_saving_path')


if '__main__' == __name__:
    tr = {}
    #tr = preprocess.load('src/training_variants', 'src/training_text') # {tr,te,tte}.pkl
    #pickle.dump(tr, open('tmp/tr.pkl', 'wb'))
    #preprocess.remove_stop_words(tr) # *.rsw.pkl
    #tr = pickle.load(open('tmp/tr.rsw.pkl', 'rb'))
    #preprocess.normalize_gene(tr) # not yet
    #preprocess.replace_text(tr, in_field='Gene', to_str=' __TARGET_GENE__ ')
    #preprocess.normalize_target_variation(tr)
    #preprocess.replace_text(tr, in_field='Variation', to_str=' __TARGET_VARIATION__ ')
    #preprocess.replace_classified_variant(tr) # not yet
    #preprocess.sentences(tr)
    #preprocess.paragraph_by_variation(tr, 0) # *.p.pkl
    tr = pickle.load(open('tmp/tr.p.pkl', 'rb'))
    te = pickle.load(open('tmp/te.p.pkl', 'rb'))
    tfidf_map = encode.tfidf_sequential(tr) # tr.ts.pkl, different length
    pickle.dump(tr, open('tmp/tr.ts.pkl', 'wb'))
    encode.tfidf_sequential(te, tfidf_map)
    pickle.dump(te, open('tmp/te.ts.pkl', 'wb'))
    sys.exit(0)
#    tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#    tf['svd'] = encoding.svd(tr['tfidf'], 50)
#    tf['d2v'] = encoding.doc2vec(tr, 50)
#    tf['tfidf'] = encoding.tfidf(tr['text'], 50)
    train(tr)

# vi:et:sw=4:ts=4
