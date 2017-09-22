#!/usr/bin/env python3

import pickle
import sys
from util import encode, preprocess

if '__main__' == __name__:
    tr = {}
    #tr = preprocess.load('src/training_variants', 'src/training_text') # tr.pkl
    #pickle.dump(tr, open('tmp/tr.pkl', 'wb'))
    #preprocess.remove_stop_words(tr) # tr.rsw.pkl
    #tr = pickle.load(open('tmp/tr.rsw.pkl', 'rb'))
    #preprocess.normalize_gene(tr)
    #preprocess.normalize_variant(tr)
    #preprocess.replace_target_gene(tr)
    #preprocess.replace_target_variation(tr, '__TARGET_VARIATION__')
    #preprocess.replace_classified_variant(tr)
    #preprocess.sentences(tr)
    #preprocess.paragraph_by_variation(tr, window_size=0) # tr.p.pkl
    tr = pickle.load(open('tmp/tr.p.pkl', 'rb'))
    #encode.tfidf_sequential(tr)
    sys.exit(0)
    pickle.dump(tr, open('tmp/tmp.pkl', 'wb'))
#    tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#    tf['svd'] = encoding.svd(tr['tfidf'], 50)
#    tf['d2v'] = encoding.doc2vec(tr, 50)
#    tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4
