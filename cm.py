#!/usr/bin/python
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np

y_pred = pickle.load(open('shuffle.pkl', 'rb'))
y_true = pickle.load(open('ytrue.pkl', 'rb'))
y_pred_post = []

i = 0
for _ in y_pred:
    max = 0
    j = 0
    max_idx = j
    for item in y_pred[i]:
        if max < item:
            max = item
            max_idx = j
            j = j + 1
    y_pred_post.append(j + 1)
    i = i + 1
y_true = [item.tolist().index(1) + 1 for item in y_true]
cm = confusion_matrix(y_true, y_pred_post)
with open('nn_cm.pkl', 'wb') as p:
    pickle.dump(cm, p)

