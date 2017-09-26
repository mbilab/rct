from datetime import datetime
from os.path import isfile
import numpy
import pickle

def field_array(data, field):
    return [d[field] for d in data]

def histogram(data):
    size = [len(d['X']) for d in data]
    hist = numpy.histogram(size, max(size), (0, max(size)))
    total = len(size)
    counted = 0
    for c, b in zip(hist[0], hist[1]): # count, bin
        if c:
            counted += c
            print("%s\t%s\t%s" % (int(b), c, counted / total))

def load(filename):
    return pickle.load(open(filename, 'rb'))

def save(data, filename, X_field=None, y_field=None, force=False):
    if isfile(filename) and not force:
        raise FileExistsError('%s existed, please delete it manually before saving' % (filename))
    if X_field:
        if y_field:
            data = [{ 'X': d[X_field], 'y': d[y_field] } for d in data]
        else:
            data = [{ 'X': d[X_field] } for d in data]
    pickle.dump(data, open(filename, 'wb'))
    print('%s saved' % (filename))

def tick(last=None):
    n = datetime.now()
    if last:
        print(n - last)
    return n

def to_csv(data, fields):
    print(','.join(fields))
    for d in data:
        print(','.join([d[f] for f in fields]))

# vi:et:sw=4:ts=4
