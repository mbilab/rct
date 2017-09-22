import os
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

#from gensim.models import Doc2Vec
#from gensim.models.doc2vec import LabeledSentence

#from keras.preprocessing.text import text_to_word_sequence

from util import field_array

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
    X = field_array(data, 'text')
    if not tfidfer:
        tfidfer = TfidfVectorizer(**kwargs)
        tfidfer.fit(X)
    tfidfed = tfidfer.transform(X)
    for d, t in zip(data, tfidfed):
        d['tfidf'] = t
    return tfidfer

def tfidf_sequential(data, model):
    for d in data:
        d['tfidf'] = []
        for word in d['text'].split():
            try:
                d['tfidf'].append(model['values'][:,model['terms'].index(word)])
            except ValueError:
                pass

def tfidf_sequential_fit(data, only_overall=True, **kwargs):
    X = field_array(data, 'text')
    X.append(' '.join(X))
    tfidfer = TfidfVectorizer(**kwargs)
    tfidfer.fit(X)
    values = tfidfer.transform(X)
    if only_overall:
        values = values[-1]
    return {
            'terms': tfidfer.get_feature_names(),
            'values': values,
            }

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

# vi:et:sw=4:ts=4
