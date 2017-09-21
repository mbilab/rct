import os

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils

from keras.preprocessing.text import text_to_word_sequence

def tfidf_SVD(df_data, l):
    vect = TfidfVectorizer()
    sentence_vectors = vect.fit_transform(df_data['Text'])
    svd = TruncatedSVD(l)
    sentence_vectors = svd.fit_transform(sentence_vectors)
    return sentence_vectors


def doc2vec(data, l, model):
    sentences = [el['text'] for el in data]
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
