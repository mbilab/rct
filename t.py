#!/usr/bin/env python3

import preprocess from util

if '__main__' == __name__:
    tr['text'] = util.load('train_variant', 'train_text', pickle='tr.pickle')
    tr['text'] = util.remove_stop_words(tr)
    tr['text'] = util.normalize_gene(tr)
    tr['text'] = util.normalize_variant(tr)
    tr['text'] = util.replace_target_gene(tr)
    tr['text'] = util.replace_target_variant(tr)
    tr['text'] = util.replace_classified_variant(tr)
    tr['text'] = util.split_text_by_variant(tr, window_size=1, unit='sentence')
    tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
    tf['tfidf'] = encoding.tfidf(tr, 50)
    tf['svd'] = encoding.svd(tr['tfidf'], 50)
    tf['d2v'] = encoding.doc2vec(tr, 50)
    tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4
