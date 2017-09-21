#!/usr/bin/env python3

from util import preprocess

if '__main__' == __name__:
#    tr['text'] = util.load('train_variant', 'train_text', pickle='tr.pickle')
#    tr['text'] = util.remove_stop_words(tr)
#    tr['text'] = util.normalize_gene(tr)
#    tr['text'] = util.normalize_variant(tr)
#    tr['text'] = util.replace_target_gene(tr)
#    tr['text'] = util.replace_target_variant(tr)
#    tr['text'] = util.replace_classified_variant(tr)
    tr = [
            { 'sentences': ['aa bb cc __SENTENCE_END__', 'bb cc dd __SENTENCE_END__'] },
            { 'sentences': ['bb cc dd __SENTENCE_END__', 'cc dd ee __SENTENCE_END__', 'dd ee ff __SENTENCE_END__'] },
            ]
    preprocess.paragraph_by_variant(tr, window_size=0, target_variant='cc')
#    tr['text'] = util.favorite(tr['text'], pickle='tr.pickle')
#    tf['tfidf'] = encoding.tfidf(tr, 50)
#    tf['svd'] = encoding.svd(tr['tfidf'], 50)
#    tf['d2v'] = encoding.doc2vec(tr, 50)
#    tf['tfidf'] = encoding.tfidf(tr['text'], 50)

# vi:et:sw=4:ts=4