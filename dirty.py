#!/usr/bin/env python3

import numpy
import pandas
from sklearn.metrics import log_loss
import sys
import pickle
from keras.models import load_model

import util
from util import encode, preprocess
from util import nn

# from xgboost import cv, DMatrix, XGBClassifier

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

    tr = util.load('tmp/tr.dsc-4.pkl')
    #nn.cnn_train(tr)
    X_val, y_val = nn.cnn_train(tr)
    #nn.rnn_train(tr)
    model = load_model('best_model_saving_path')
    prob = model.predict(X_val)
    with open('shuffle.pkl', 'wb') as p:
        pickle.dump(prob, p)
    sys.exit(0)

    #tr = util.load('ignore/stage_1_tmp/tr.pkl')
    #tte = util.load('ignore/stage_1_tmp/tte.pkl')
    #tt = tr + tte
    #encode.tfidf(tt)
    #tt = util.load('tmp1/all.t.pkl')
    #encode.svd(tt, n_components=64)
    #util.save(tt, 'tmp1/all.s64.pkl')

    #predict('tmp1/tr.dsc-4.h5', 'tmp1/tte.dsc-4.pkl', 'tmp1/tte.pkl')

    tt = util.load('tmp1/all.s64.pkl')
    tt = tt[:3321]
    tr = pandas.read_csv('ignore/stage_1_tmp/tr.dsc-4.csv').values
    tr = numpy.delete(tr, [0, 1] + list(range(66, 75)), 1)
    #tr = numpy.delete(tr, [0, 1], 1)
    for d, nn in zip(tt, tr):
        #d['X'] = numpy.hstack([d['svd'], nn])
        #d['X'] = numpy.hstack([d['svd']])
        d['X'] = numpy.hstack([nn])
        d['y'] = int(d.pop('Class')) - 1
    X = numpy.vstack(util.field_array(tt, 'X'))
    y = util.field_array(tt, 'y')
    #out = cv({
    #    'learning_rate': 0.1,
    #    'max_depth': 4,
    #    'n_estimators': 100,
    #    'objective': 'multi:softprob',
    #    'num_class': 9
    #    }, DMatrix(X, y), seed=0)
    #print(out)
    #sys.exit(0)

    model = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=100, objective='multi:softprob', seed=0)
    model.fit(X, y)

    tt = util.load('tmp1/all.s64.pkl')
    tt = tt[3321:]
    tr = pandas.read_csv('ignore/stage_1_tmp/tte.dsc-4.csv').values
    tr = numpy.delete(tr, [0, 1] + list(range(66, 75)), 1)
    #tr = numpy.delete(tr, [0, 1], 1)
    for d, nn in zip(tt, tr):
        #d['X'] = numpy.hstack([d['svd'], nn])
        #d['X'] = numpy.hstack([d['svd']])
        d['X'] = numpy.hstack([nn])
        d['y'] = int(d.pop('Class')) - 1
    X = numpy.vstack(util.field_array(tt, 'X'))
    y = util.field_array(tt, 'y')
    proba = model.predict_proba(X)
    print(log_loss(y, proba))

#   preprocess.normalize_gene(tr) # not yet
#   preprocess.replace_text(tr, in_field='Gene', to_str=' __TARGET_GENE__ ')
#   preprocess.replace_classified_variant(tr) # not yet
#   tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#   tf['svd'] = encoding.svd(tr['tfidf'], 50)
#   tf['d2v'] = encoding.doc2vec(tr, 50)
#   tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4
