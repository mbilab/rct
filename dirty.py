#!/usr/bin/env python

import numpy
from keras.models import load_model
import pandas
import pickle
from random import Random
from sklearn.metrics import confusion_matrix, log_loss
import sys
from xgboost import XGBClassifier

import util
from util import encode, preprocess
from util import nn

# from xgboost import cv, DMatrix, XGBClassifier

def xgb_model(model_filename):
    tt = util.load('tmp/all.s64.pkl')[:3321]
    for d in tt:
        d['X'] = d.pop('svd')
        d['y'] = int(d.pop('Class'))
    X, y, Xv, yv = preprocess.format_data(tt, 0.2)
    model = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=1000, objective='multi:softprob', seed=0)
    X = numpy.vstack(X)
    model.fit(X, y)
    return model

def merge_nn_and_xgb():
    te = preprocess.subset('tmp/te.dsc-4.pkl', 'tmp/te.pkl')
    nn_prob = nn.predict('tmp/tr.dsc-4.h5', te)

    model = xgb_model('tmp/all.s64.pkl')
    tt = util.load('tmp1/all.s64.pkl')[3321:]
    X = util.field_array(tt, 'svd')
    y = util.field_array(tt, 'Class')
    xgb_prob = model.predict_proba(X)

    prob = []
    for n, x in zip(nn_prob, xgb_prob):
        if (0 == x.tolist().index(max(x))) and (0 != n.tolist().index(max(n))) and (6 != n.tolist().index(max(n))):
            prob.append(n)
        else:
            prob.append(x)
    print(log_loss(y, prob))
    pred = [v.tolist().index(max(v)) for v in prob]
    m = confusion_matrix(y, pred)
    print(m)

def evaluate_nn():
    tr = util.load('tmp/tr.dsc-4.pkl')
    X, y, Xv, yv = nn.format_data(tr, y_to_categorical=False)
    model = load_model('tmp/tr.dsc-4.h5')
    util.evaluate(model, Xv, yv)

def evaluate_xgb():
    tt = util.load('tmp/all.s64.pkl') # svd 64d
    tt = tt[:3321]
    for d in tt:
        d['X'] = d.pop('svd')
        d['y'] = int(d.pop('Class'))
    X, y, Xv, yv = preprocess.format_data(tt, 0.2)
    model = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=1000, objective='multi:softprob', seed=0)
    X = numpy.vstack(X)
    model.fit(X, y)
    util.evaluate(model, Xv, yv)

if '__main__' == __name__:

    #util.s('tmp/tr', 'tmp/te') # *.s.pkl
    #util.swv('tmp/te.rsw.pkl') # *.swv.pkl, 0:01:10.617185
    #util.s2ds('tmp/tr.s.pkl', 0, True, 1e-4, 'tmp/te.s.pkl') # *.dsc*.pkl, 0:07:59.554650

    #evaluate_nn()
    #evaluate_xgb()

    #te = preprocess.subset('tmp/te.dsc-4.pkl', 'tmp/te.pkl')
    #nn.predict('tmp/tr.dsc-4.h5', te)

def tfidf_value():
    return
    #encode.tfidf_sequential(tr, tsm) # *(.c).ts.pkl, 0:08:12.010365
    #util.save(tr, 'tmp/tr.ts.pkl', 'tfidf', 'Class')

    #tr = util.load('tmp/tmp2.pkl')
    #encode.sparse_clean(tr, 0.5) # *(.c).ts.sc*.pkl, 0:07:57.079560
    #util.save(tr, 'tmp/tmp.pkl')

def two_stage():
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
    prob = model.predict_proba(X)
    print(log_loss(y, prob))

# vi:et:sw=4:ts=4
