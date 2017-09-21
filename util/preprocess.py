from nltk import sent_tokenize
from nltk.corpus import stopwords
from os.path import isfile
import pandas
import pickle
import re

def find_pickle(filename):
    for ext in ['', '.pickle', 'pkl']:
        path = filename + ext
        if (isfile(path)):
            return path
    return None

def load(variant_file, text_file, pickle_file):
    path = find_pickle(pickle_file)

    if path:
        return pickle.load(open(path, 'rb'))

    variant = pandas.read_csv(variant_file)
    text = pandas.read_csv(text_file, sep = '\|\|', header = None, skiprows = 1, names = ['ID', 'text'])

    tr = []

    for ID in variant['ID'].values:
        item = { info: variant[info][ID] for info in variant.columns.values }
        item['Text'] = text['text'][ID]
        tr.append(item)

    pickle.dump(tr, open(pickle_file, 'wb'))

    return tr

def paragraph_by_variant(data, window_size=0, unit='sentence', pickle_file=None, target_variant='__TARGET_VARIANT__', paragraph_end=' __PARAGRAPH_END__ '):
    if pickle_file:
        path = find_pickle(pickle_file)
        if path:
            return pickle.load(open(path, 'rb'))

    for d in data:
        d['text'] = ''
        s = d['sentences']
        for i in range(len(s)):
            if -1 != s[i].find(target_variant):
                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(s))):
                    d['text'] += s[j]
                d['text'] += paragraph_end
        print(d['text'])

def remove_stopwords(tr):
    stopwords_list = stopwords.words('english')
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s+')
    for data in tr :
        if 'Text' in data :
            data['Text'] = pattern.sub("", data['Text'])
    return None

# vi:et:sw=4:ts=4
