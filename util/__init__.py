from datetime import datetime
from os.path import isfile
import numpy
import pickle
import re

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

# sentence to dummy sequence
def s2ds(filename, paragraph_size=0, concatenate_by_class=True, tfidf_tolerance=0, test_filename=None): # 0:07:59.554650
    t = tick()
    data = load(filename)
    preprocess.paragraph_by_variation(data, paragraph_size) # *.pbvw*.pkl, 0:00:00.772103
    #util.save(tr, filename.replace('.pkl', ".pbvw%s.pkl" % (paragraph_size)))
    c = preprocess.concatenate(data) if concatenate_by_class else data # 0:00:01.121521
    tsm = encode.tfidf_sequential_model(c) # 0:00:03.516558
    encode.dummy_sequence(data, tsm, tfidf_tolerance) # *.ds.pkl, 0:04:59.167635
    c = 'c' if concatenate_by_class else ''
    save(data, filename.replace('.pkl', '.ds%s%s.pkl' % (c, tfidf_tolerance)), 'dummy', 'Class')
    if test_filename:
        filename = test_filename
        data = load(filename)
        encode.dummy_sequence(data, tsm, tfidf_tolerance)
        c = 'c' if concatenate_by_class else ''
        save(data, filename.replace('.pkl', '.ds%s%s.pkl' % (c, tfidf_tolerance)), 'dummy', 'Class')
    t = tick(t, 's2ds')

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

# sentence with variation
def swv(filename): # 0:01:10.617185
    t = tick()
    data = load(filename)
    preprocess.normalize_target_variation(data) # 0:00:11.750795
    preprocess.replace_text(data, in_field='Variation', to_str=' __TARGET_VARIATION__ ') # 0:00:00.348791
    preprocess.sentences(data, None) # *.s.pkl, 0:01:16.074815
    preprocess.paragraph_by_variation(data, use_first_sentence=False, paragraph_end=' ') # *.pbvw*.pkl, 0:00:00.772103
    for d in data:
        d['Text'] = re.sub(r'__TARGET_VARIATION__', d['Variation'], d['Text']).rstrip()
    #to_csv(data, ['Class', 'Gene', 'Variation', 'Text'])
    to_csv(data, ['Gene', 'Variation', 'Text'])
    t = tick(t, 'swv')

def tick(last=None, name=None):
    n = datetime.now()
    if last:
        if name:
            print('%s: %s' % (name, n - last))
        else:
            print(n - last)
    return n

def to_csv(data, fields):
    print(','.join(fields))
    for d in data:
        print(','.join([str(d[f]) for f in fields]))

# vi:et:sw=4:ts=4
