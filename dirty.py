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

def merge_nn_and_xgb():
    model = load_model('1.tr.dsc-4.h5')
    tte = util.load('ignore/stage_1_tmp/tte.dsc-4.pkl')
    X, y = nn.format_data(tte, False, 0)
    prob = model.predict(X)
    nn_prob = prob
    print(log_loss(y, prob))
    pred = [v.tolist().index(max(v)) for v in prob]
    m = confusion_matrix(y, pred)
    #print(m)

    tt = util.load('tmp1/all.s64.pkl')
    tt = tt[:3321]
    for d in tt:
        d['X'] = d.pop('svd')
        d['y'] = int(d.pop('Class'))
    X, y, Xv, yv = preprocess.format_data(tt, 0.2)
    model = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=1000, objective='multi:softprob', seed=0)
    X = numpy.vstack(X)
    model.fit(X, y)
    tt = util.load('tmp1/all.s64.pkl')
    tt = tt[3321:]
    for d in tt:
        d['X'] = d.pop('svd')
        d['y'] = int(d.pop('Class'))
    X, y = preprocess.format_data(tt)
    X = numpy.vstack(X)
    prob = model.predict_proba(X)
    xgb_prob = prob
    print(log_loss(y, prob))
    pred = model.predict(Xv)
    m = confusion_matrix(yv, pred)
    #print(m)
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

def nn_val_cm():
    #tr = util.load('ignore/stage_1_tmp/tr.dsc-4.pkl')
    tr = util.load('tmp/tr.dsc-4.pkl')
    X, y, Xv, yv = nn.format_data(tr, False)
    model = load_model('2.tr.dsc-4.h5')
    prob = model.predict(Xv)
    print(log_loss(yv, prob))
    pred = [v.tolist().index(max(v)) for v in prob]
    m = confusion_matrix(yv, pred)
    print(m)

def xgb_val_cm():
    tt = util.load('tmp1/all.s64.pkl')
    tt = tt[:3321]
    for d in tt:
        d['X'] = d.pop('svd')
        d['y'] = int(d.pop('Class'))
    X, y, Xv, yv = preprocess.format_data(tt, 0.2)
    model = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=1000, objective='multi:softprob', seed=0)
    X = numpy.vstack(X)
    model.fit(X, y)
    prob = model.predict_proba(Xv)
    print(log_loss(yv, prob))
    pred = model.predict(Xv)
    m = confusion_matrix(yv, pred)
    print(m)

if '__main__' == __name__:

    #util.s('tmp/tr', 'tmp/te') # *.s.pkl
    #util.swv('tmp/te.rsw.pkl') # *.swv.pkl, 0:01:10.617185

    #tr = util.s2ds('tmp/tr.s.pkl', 0, True, 1e-4, 'tmp/te.s.pkl') # 0:07:59.554650
    #sys.exit(0)

    #encode.tfidf_sequential(tr, tsm) # *(.c).ts.pkl, 0:08:12.010365
    #util.save(tr, 'tmp/tr.ts.pkl', 'tfidf', 'Class')

    #tr = util.load('tmp/tmp2.pkl')
    #encode.sparse_clean(tr, 0.5) # *(.c).ts.sc*.pkl, 0:07:57.079560
    #util.save(tr, 'tmp/tmp.pkl')

    #tr = util.load('tmp/te.pkl')
    #tte = util.load('src/stage_1/trueTstTotal.pkl')
    #tr = preprocess.subset(tr, tte) # tte(.c).ts.pkl, 0:00:00.180169
    #encode.tfidf_sequential(tr, tsm)
    #encode.dummy_sequential(tr, tsm)
    #util.save(tr, 'tmp1/tte.pkl')

    #tr = util.load('tmp/tr.dsc-4.pkl')
    #util.histogram(tr)

    #tr = util.load('ignore/stage_1_tmp/tr.dsc-4.pkl')
    #nn.cnn_train(tr)
    #Xv, yv = nn.cnn2_train(tr)
    #nn.rnn_train(tr)
    #print(log_loss(yv, prob))

    #nn_val_cm()
    #xgb_val_cm()

    tr = util.load('tmp/te.dsc-4.pkl')
    for d in tr:
        d['y'] = 0
    X, y = nn.format_data(tr, False, 0)
    model = load_model('2.tr.dsc-4.h5')
    prob = model.predict(X)
    tr = util.load('tmp/te.pkl')
    print(','.join(
        ['Gene', 'Variation'] +
        ['O'+str(i+1) for i in range(prob.shape[1])]))
    for i in range(len(prob)):
        d = tr[i]
        print(','.join([str(v) for v in [
            d['Gene'], d['Variation']] +
            list(prob[i])]))
    #print(log_loss(y, prob))
    #pred = [v.tolist().index(max(v)) for v in prob]
    #m = confusion_matrix(yv, pred)
    #print(m)
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
    prob = model.predict_proba(X)
    print(log_loss(y, prob))

#   preprocess.normalize_gene(tr) # not yet
#   preprocess.replace_text(tr, in_field='Gene', to_str=' __TARGET_GENE__ ')
#   preprocess.replace_classified_variant(tr) # not yet
#   tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#   tf['svd'] = encoding.svd(tr['tfidf'], 50)
#   tf['d2v'] = encoding.doc2vec(tr, 50)
#   tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4
