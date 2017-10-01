from datetime import datetime
from os.path import isfile
import numpy
import pickle
import re

def evaluate(model, X, y):
    prob = model.predict(Xv)
    print(log_loss(y, prob))
    pred = [v.tolist().index(max(v)) for v in prob]
    m = confusion_matrix(y, pred)
    print(m)

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

# sentence
def s(*filenames):
    for f in filenames:
        data = preprocess.load(f+'.csv') # *.pkl, 0:00:01.650411
        util.save(data, f+'.pkl')

        preprocess.remove_stop_words(data) # *.rsw.pkl, 0:00:46.640691
        util.save(data, f+'.rsw.pkl')

        #data = util.load(f+'.rsw.pkl')
        preprocess.normalize_target_variation(data) # 0:00:11.750795
        preprocess.replace_text(data, in_field='Variation', to_str=' __TARGET_VARIATION__ ') # 0:00:00.348791
        preprocess.replace_text(data, in_field='variation position', to_str=' __TARGET_VARIATION_POSITION__ ') # 0:00:00.348791
        preprocess.sentences(data) # *.s.pkl, 0:01:16.074815
        util.save(data, f+'.s.pkl')

# sentence to dummy sequence
def s2ds(filename, paragraph_size=0, remove_samples=False, concatenate_by_class=True, tfidf_tolerance=0, test_filename=None): # 0:07:59.554650
    t = tick()
    data = load(filename)
    preprocess.paragraph_by_variation(data, paragraph_size) # *.pbvw*.pkl, 0:00:00.772103
    if remove_samples:
        data = [d for d in data if d['Text']]
    #util.save(tr, filename.replace('.pkl', ".pbvw%s.pkl" % (paragraph_size)))
    c = preprocess.concatenate(data) if concatenate_by_class else data # 0:00:01.121521
    tsm = encode.tfidf_sequential_model(c) # 0:00:03.516558
    encode.dummy_sequence(data, tsm, tfidf_tolerance) # *.ds.pkl, 0:04:59.167635
    #util.histogram(data)
    c = 'c' if concatenate_by_class else ''
    #save(data, filename.replace('.pkl', '.ds%s%s.pkl' % (c, tfidf_tolerance)), 'dummy', 'Class')
    if test_filename:
        filename = test_filename
        data = load(filename)
        encode.dummy_sequence(data, tsm, tfidf_tolerance)
        c = 'c' if concatenate_by_class else ''
        #save(data, filename.replace('.pkl', '.ds%s%s.pkl' % (c, tfidf_tolerance)), 'dummy', 'Class')
        save(data, filename.replace('.pkl', '.ds%s%s.pkl' % (c, tfidf_tolerance)), 'dummy')
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
