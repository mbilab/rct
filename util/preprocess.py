#!/usr/bin/env python3

from nltk import sent_tokenize
from nltk.corpus import stopwords
import json
import pandas
import pickle
import re

from util import field_array
from util.variation import Variation

def concatenate(data, by_field='Class'):
    fields = field_array(data, by_field)
    fields = sorted(set(fields))
    field_to_index = { v: i for i, v in enumerate(fields) }
    c = [{ 'text': '' } for v in fields]
    for d in data:
        index = field_to_index[d[by_field]]
        c[index]['text'] += d['text'] + ' '
    return c

def load(variant_filename, text_filename):
    variant = pandas.read_csv(variant_filename)
    text = pandas.read_csv(text_filename, sep='\|\|', header=None, skiprows=1, names=['ID', 'text'])

    data = []
    for ID in variant['ID'].values:
        d = { info: variant[info][ID] for info in variant.columns.values }
        d['text'] = text['text'][ID]
        data.append(d)

    return data

def normalize_target_variation(data):
    aa_alias = json.load(open('one2many.json'))
    for d in data:
        v = Variation(d['Variation'])
        if 'point' == v.type:
            starts = [v.start_amino] + aa_alias[v.start_amino.upper()]
            if '*' == v.end_amino:
                aliases = ['%s%sX'  % (s, v.pos) for s in starts]
            elif '' == v.end_amino:
                aliases = ["%s%s"   % (s, v.pos) for s in starts]
            else:
                aliases = ["%s%s%s" % (s, v.pos, e) for s in starts for e in [v.end_amino] + aa_alias[v.end_amino.upper()]]
            d['text'] = re.sub('%s' % '|'.join(aliases), v.var, d['text'], flags=re.IGNORECASE)

def paragraph_by_variation(data, window_size=0, unit='sentence', target_variation='__TARGET_VARIATION__', paragraph_end=' __PARAGRAPH_END__ '):
    for d in data:
        d['text'] = ''
        s = d['sentences']
        for i in range(len(s)):
            if -1 != s[i].find(target_variation):
                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(s))):
                    d['text'] += s[j]
                d['text'] += paragraph_end
        if '' == d['text']:
            #d['text'] = 'No target variation found.'
            d['text'] = d['sentences'][0] + paragraph_end
        #d['text'] = re.sub(r'__TARGET_VARIATION__', d['Variation'], d['text']).rstrip()

def remove_stop_words(data, pickle_filename=None):
    if pickle_filename:
        path = find_pickle(pickle_filename)
        if path:
            return pickle.load(open(path, 'rb'))

    stopwords_list = stopwords.words('english')
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s+')
    for d in data:
        d['text'] = pattern.sub('', d['text'])

    if pickle_filename:
        pickle.dump(data, open(pickle_filename, 'wb'))

def replace_text(data, in_field=None, to_str=None):
    for d in data:
        d['text'] = re.sub(d[in_field], to_str, d['text'])

def sentences(data, sentence_end=' __SENTENCE_END__ '):
    for d in data:
        d['sentences'] = []
        text = re.sub(r'\s\.([A-Z]\w+)', r'\s \1', d['text'])
        if sentence_end:
            d['sentences'] = [re.sub(r'\.?$', sentence_end, s).rstrip() for s in sent_tokenize(text)]
        else:
            d['sentences'] = [s.rstrip() for s in sent_tokenize(text)]

def subset(data, sub_data, sub_fields=['Class'], key_fields=['Gene', 'Variation']):
    keys = []
    for d in sub_data:
        key = '__'.join([d[f] for f in key_fields])
        keys.append(key)
    new_data = []
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

def normalize_gene(data):
    gene_alias_dict = pickle.load(open('./gene_alias.regex.pkl', 'rb'))
    #print(gene_alias_dict)
    return
    for item in tr:
        text = item['text']
        if not re.match('ID', text):
            gene = text.split(',')[1]
            if gene in gene_alias_dict:
                text = re.sub(gene_alias_dict[gene], '$_TARGET_GENE_$', text)
        item['text'] = text

def replace_target_gene(tr):
    gene_alias_dict = pickle.load(open('./gene_alias.regex.pkl', 'r'))
    for item in tr:
        text = item['text']
        if not re.match('ID', text):
            gene = text.split(',')[1]
            if gene in gene_alias_dict:
                text = re.sub(gene_alias_dict[gene], '$_TARGET_GENE_$', text)
        item['text'] = text

def replace_classified_variant(tr):
    answer_dict = pickle.load(open('./answer_dict.pkl', 'r'))
    for item in tr:
        text = item['text']
        if not re.match('ID', text):
            gene, variation = variant.split(',')[1:3]
            if gene in answer_dict:
                for variation in answer_dict[gene]:
                    answer = answer_dict[gene][variation]
                    text = convertVariation(text, variation, answer)
        item['text'] = text

# vi:et:sw=4:ts=4
