import os
import pickle

import preprocess.py

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

from keras.preprocessing.text import text_to_word_sequence


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


def tfidf(data, tfidfer=None, pickle_file=None, **kwargs):
    if pickle_file:
        path = preprocess.find_pickle(pickle_file)
        if path:
            return pickle.load(open(path, 'rb'))
    X = [d['text'] for d in data]
    if not tfidfer:
        tfidfer = TfidfVectorizer(**kwargs)
        tfidfer.fit(X)
    tfidfed = tfidfer.transform(X)
    for d, t in zip(data, tfidfed):
        d['tfidf'] = t
    return tfidfer


def tfidf_sequential(data, tfidfer=None, pickle_file=None, **kwargs):
    if pickle_file:
        path = preprocess.find_pickle(pickle_file)
        if path:
            return pickle.load(open(path, 'rb'))
    X = [d['text'] for d in data]
    # cat all text and append it to X
    if not tfidfer:
        tfidfer = TfidfVectorizer(**kwargs)
        tfidfer.fit(X)
    tfidfed = tfidfer.transform(X)
    names = tfidfer.get_feature_names()
    # now we use the tfidf of the last element in X
    for d in data:
        d['tfidf'] = []
        for word in d['text'].split():
            if word in names:
                d['tfidf'].append(tfidfed[names.index(word)])
    return tfidfer


def tfidf_SVD(data, l):
    sentences = [el['text'] for el in data]
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
        label_sentences.append(LabeledSentence(text_to_word_sequence(row['text']), ['Text' + '_%s' % str(index)]))
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
