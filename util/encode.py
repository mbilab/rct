import os
import pickle

from scipy.sparse.csr import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

#from gensim.models import Doc2Vec
#from gensim.models.doc2vec import LabeledSentence

#from keras.preprocessing.text import text_to_word_sequence

from util import field_array

def dummy_sequence(data, term_value, tolerance=0):
    terms = ['']
    for i in range(len(term_value['terms'])):
        if term_value['values'][0,i] >= tolerance:
            terms.append(term_value['terms'][i])
    for d in data:
        d['dummy'] = []
        for term in term_value['tokenizer'](d['Text'].lower()):
            try:
                i = terms.index(term)
                d['dummy'].append(i)
            except ValueError:
                pass

def sparse_clean(data, tolerance=0.01, field='X'):
    o = 0
    n = 0
    if isinstance(data[0][field][0], csr_matrix):
        for d in data:
            o += len(d[field])
            d[field] = [v for v in d[field] if max(v) >= tolerance]
            n += len(d[field])
    else:
        for d in data:
            o += len(d[field])
            d[field] = [v for v in d[field] if v >= tolerance]
            n += len(d[field])
    print('sparse clean (<%s) from %s to %s terms' % (tolerance, o, n))

def svd(data, svder=None, input_field='tfidf', pickle_file=None, **kwargs):
    if pickle_file:
        path = preprocess.find_pickle(pickle_file)
        if path:
            return pickle.load(open(path, 'rb'))
    X = [d[input_field] for d in data]
    if not svder:
        svder = TruncatedSVD(**kwargs)
        svder.fit(X)
    svded = svder.transform(X)
    for d, s in zip(data, svded):
        d['svd'] = s
    return svder

def tfidf(data, tfidfer=None, **kwargs):
    X = field_array(data, 'Text')
    if not tfidfer:
        tfidfer = TfidfVectorizer(**kwargs)
        tfidfer.fit(X)
    tfidfed = tfidfer.transform(X)
    for d, t in zip(data, tfidfed):
        d['tfidf'] = t
    return tfidfer

def tfidf_sequential(data, term_value):
    for d in data:
        d['tfidf'] = []
        for term in term_value['tokenizer'](d['Text'].lower()):
            try:
                i = term_value['terms'].index(term)
                d['tfidf'].append(term_value['values'][:,i])
            except ValueError:
                pass

def tfidf_sequential_model(data, only_overall=True, **kwargs):
    X = field_array(data, 'Text')
    X.append(' '.join(X))
    tfidfer = TfidfVectorizer(**kwargs)
    tfidfer.fit(X)
    values = tfidfer.transform(X)
    if only_overall:
        values = values[-1]
    terms = tfidfer.get_feature_names()
    #n = len(terms)
    #s = sorted(range(n), key=lambda k: values[0,k], reverse=True)
    #for i in range(n):
    #    print('%s\t%s\t%s' % (terms[s[i]], values[0,s[i]], (i+1) / n))
    return {
            'terms': terms,
            'tokenizer': tfidfer.build_tokenizer(),
            'values': values,
            }

def tfidf_SVD(data, l):
    sentences = [el['Text'] for el in data]
    vect = TfidfVectorizer()
    sentence_vectors = vect.fit_transform(sentences)
    svd = TruncatedSVD(l)
    sentence_vectors = svd.fit_transform(sentence_vectors)
    index = 0
    for _ in data:
        data[index]['tfidf_SVD'] = sentence_vectors[index]
        index = index + 1

def doc2vec(data, l, model):
    label_sentences = []
    index = 0
    for row in data:
        label_sentences.append(LabeledSentence(text_to_word_sequence(row['Text']), ['Text' + '_%s' % str(index)]))
        index = index + 1

    if os.path.isfile(model):
        text_model = Doc2Vec.load(model)
    else:
        text_model = Doc2Vec(min_count=1, window=5, size=l, sample=1e-4,
                             negative=5, workers=4, iter=5, seed=1)
        text_model.build_vocab(label_sentences)
        text_model.train(label_sentences, total_examples=text_model.corpus_count,
                         epochs=text_model.iter)
        text_model.save(model)

    index = 0
    for _ in data:
        data[index]['d2v'] = text_model.docvecs['Text_' + str(index)]
        index = index + 1
    # all_encode = [text_model.docvecs['Text_' + str(i)] for i in range(l)]
    # return all_encode

# vi:et:sw=4:ts=4
