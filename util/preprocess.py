#!/usr/bin/env python3

from nltk import sent_tokenize
from nltk.corpus import stopwords
import json
import pandas
import pickle
from random import Random
import re

import util
from util.variation import Variation

def concatenate(data, by_field='Class'):
    fields = util.field_array(data, by_field)
    fields = sorted(set(fields))
    field_to_index = { v: i for i, v in enumerate(fields) }
    c = [{ 'Text': '' } for v in fields]
    for d in data:
        index = field_to_index[d[by_field]]
        c[index]['Text'] += d['Text'] + ' '
    return c

def format_data(data, validation_split=0, seed=0):
    Random(seed).shuffle(data)
    X = util.field_array(data, 'X')
    y = [y-1 for y in util.field_array(data, 'y')]

    if validation_split:
        split = int(validation_split * len(y))
        return X[split:], y[split:], X[:split], y[:split]
    return X, y

def load(filename):
    return pandas.read_csv(filename).to_dict('records')

def normalize_target_variation(data):
    aa_alias = json.load(open('src/aa_alias.json'))
    for d in data:
        v = Variation(d['Variation'])
        if 'point' == v.type:
            v['variation position'] = v.pos
            starts = [v.start_amino] + aa_alias[v.start_amino.upper()]
            if '*' == v.end_amino:
                aliases = ['%s%sX'  % (s, v.pos) for s in starts]
            elif '' == v.end_amino:
                aliases = ["%s%s"   % (s, v.pos) for s in starts]
            else:
                aliases = ["%s%s%s" % (s, v.pos, e) for s in starts for e in [v.end_amino] + aa_alias[v.end_amino.upper()]]
            d['Text'] = re.sub('%s' % '|'.join(aliases), v.var, d['Text'], flags=re.IGNORECASE)

def paragraph_by_regex(
        data,
        window_size=0,
        unit='sentence',
        not_found='first sentence',
        use_first_sentence=True,
        target_regex=r'__TARGET_VARIATION__|__TARGET_VARIATION_POSITION__',
        paragraph_end=' __PARAGRAPH_END__ '):
    for d in data:
        d['Text'] = ''
        s = d['sentences']
        for i in range(len(s)):
            if re.search(target_regex, s[i]):
                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(s))):
                    d['Text'] += s[j]
                d['Text'] += paragraph_end
        if '' == d['Text']:
            if 'first sentence' == not_found:
                d['Text'] = d['sentences'][0] + paragraph_end # use the first sentence instead
            elif 'full text' == not_found:
                d['Text'] = paragraph_end.join(d['sentences'])
            else:
                d['Text'] = 'No target variation found.'

def remove_stop_words(data):
    stopwords_list = stopwords.words('english')
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s+')
    for d in data:
        d['Text'] = pattern.sub('', d['Text'])

def replace_text(data, in_field=None, to_str=None):
    for d in data:
        if d[in_field]:
            d['Text'] = re.sub(d[in_field], to_str, d['Text'])

def sentences(data, sentence_end=' __SENTENCE_END__ '):
    for d in data:
        d['sentences'] = []
        text = re.sub(r'\s+([\.\!\?])([A-Z]\w+)', r'\1 \2', d['Text'])
        if sentence_end:
            d['sentences'] = [re.sub(r'[\.\!\?]?$', sentence_end, s).rstrip() for s in sent_tokenize(text)]
        else:
            d['sentences'] = [s.rstrip() for s in sent_tokenize(text)]

def subset(data, sub_data, sub_fields=['Class'], key_fields=['Gene', 'Variation']):
    keys = []
    if isinstance(sub_data, str):
        sub_data = load(sub_data)
    for d in sub_data:
        key = '__'.join([d[f] for f in key_fields])
        keys.append(key)
    new_data = []
    if isinstance(data, str):
        data = load(data)
    for d in data:
        key = '__'.join([d[f] for f in key_fields])
        try:
            i = keys.index(key)
            for f in sub_fields:
                d[f] = sub_data[i][f]
            new_data.append(d)
        except:
            pass
    return new_data

################################################################################

def replace_classified_variant(tr):
    answer_dict = pickle.load(open('./answer_dict.pkl', 'r'))
    for item in tr:
        text = item['Text']
        if not re.match('ID', text):
            gene, variation = variant.split(',')[1:3]
            if gene in answer_dict:
                for variation in answer_dict[gene]:
                    answer = answer_dict[gene][variation]
                    text = convertVariation(text, variation, answer)
        item['Text'] = text

# vi:et:sw=4:ts=4
