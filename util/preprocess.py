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

def replace_target_gene(tr):
    gene_alias_dict = pickle.load(open('./gene_alias.regex.pkl', 'r'))
    for item in tr:
        text = item['text']
        if not re.match('ID', text):
            gene = text.split(',')[1]
            if gene in gene_alias_dict:
                text = re.sub(gene_alias_dict[gene], '$_TARGET_GENE_$', text)
        item['text'] = text

def convertVariation(raw_text, variation, alternate_string):
    amino_acid_dict = pickle.load(open('./amino_acid_dict.pkl', 'r'))
    variation_info = re.findall('^([A-Za-z])(\d+)([A-Za-z])$', variation)

    if variation_info:
        original_amino, position, final_amino = variation_info[0]

        variation_list = []
        for o_amino in amino_acid_dict[original_amino]:
            for f_amino in amino_acid_dict[final_amino]:
                variation_list.append(o_amino + position + f_amino)

        variation = '(%s)' % '|'.join(variation_list)

    else:
        variation = re.escape(variation)

    return re.sub(variation, alternate_string, raw_text)

def replace_target_variant(tr):
    for item in tr:
        text = item['text']
        if not re.match('ID', text):
            variation = text.split(',')[3]
            text = convertVariation(text, variation, '$_TARGET_VARIATION_$')
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
