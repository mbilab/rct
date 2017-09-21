#!/usr/bin/env python3

import pickle
import sys
from util import preprocess

if '__main__' == __name__:
    tr = {}
    #tr = preprocess.load('src/training_variants', 'src/training_text')
    #pickle.dump(tr, open('tmp/tr.pkl', 'wb'))
    #preprocess.remove_stop_words(tr)
    tr = pickle.load(open('tmp/tmp.pkl', 'rb'))
    sys.exit(0)
    #preprocess.normalize_gene(tr)
    #preprocess.normalize_variant(tr)
    preprocess.replace_target_gene(tr)
    preprocess.replace_target_variant(tr)
    preprocess.replace_classified_variant(tr)
    tr = [
            { 'sentences': ['aa bb cc __SENTENCE_END__', 'bb cc dd __SENTENCE_END__'] },
            { 'sentences': ['bb cc dd __SENTENCE_END__', 'cc dd ee __SENTENCE_END__', 'dd ee ff __SENTENCE_END__'] },
            ]
    preprocess.paragraph_by_variant(tr, window_size=0, target_variant='cc')
    print(preprocess.field_array(tr, 'text'))
#    tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#    tf['tfidf'] = encoding.tfidf(tr, 50)
#    tf['svd'] = encoding.svd(tr['tfidf'], 50)
#    tf['d2v'] = encoding.doc2vec(tr, 50)
#    tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4
