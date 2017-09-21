import os

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils


def tfidf_SVD(df_data, l):
    vect = TfidfVectorizer()
    sentence_vectors = vect.fit_transform(df_data['Text'])
    svd = TruncatedSVD(l)
    sentence_vectors = svd.fit_transform(sentence_vectors)
    return sentence_vectors


def d2v(df_data, l, model):
    sentences = []
    for index, row in df_data['Text'].iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), \
                         ['Text' + '_%s' % str(index)]))

    if os.path.isfile(model):
        text_model = Doc2Vec.load(model)
    else:
        text_model = Doc2Vec(min_count=1, window=5, size=l, sample=1e-4, \
                             negative=5, workers=4, iter=5, seed=1)
        text_model.build_vocab(sentences)
        text_model.train(sentences, total_examples=text_model.corpus_count, \
                         epochs=text_model.iter)
        text_model.save(model)

    all_encode = [text_model.docvecs['Text_' + str(i)] for i in range(l)]
    return all_encode
