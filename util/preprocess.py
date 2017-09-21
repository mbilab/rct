from nltk import sent_tokenize
from os.path import isfile
import re
import nltk
from nltk.corpus import stopwords

def remove_stopwords(tr):
    stopwords_list = stopwords.words('english')
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s+')
    for data in tr :
        if 'Text' in data :
            data['Text'] = pattern.sub("", data['Text'])
    return None

def find_pickle(filename):
    for ext in ['', '.pickle', 'pkl']:
        path = filename + ext
        if (isfile(path)):
            return path
    return None

def split_text_by_variant(data, window_size=1, unit='sentence', pickle_file=None):
    if pickle_file:
        path = find_pickle(pickle_file)
        if path:
            return pickle.load(open(path, 'rb'))

    for d in data:
        sentences = sent_tokenize(d['text'].rstrip())
        '''
        for s in sent_tokenize(d['text'].rstrip()):
            while True:
                stc_end = re.search('\s\.[A-Z]\w+', _stc)
                        stc = _stc[:stc_end.start() + 2] if stc_end else _stc

                        if len(re.findall(opt['variation']['target_text'], stc)):
                            stcid = len(total_sentences)
                            p_start_id = stcid - opt['window_size'] / 2
                            p_end_id = stcid + opt['window_size'] / 2 + 1

                            reserved_paragraphs.append([p_start_id, p_end_id])

                        total_sentences.append(stc)

                        if not stc_end:
                            break

                        _stc = _stc[stc_end.start() + 2:]

                if not len(reserved_paragraphs):
                    reserved_paragraphs.append([0, min(len(total_sentences), opt['window_size'])])

                row = row_id + '||'

                for p in reserved_paragraphs:
                    for stcid in range(p[0], p[1]):
                        if stcid < 0 or stcid >= len(total_sentences):
                            continue

                        row += total_sentences[stcid]

                        if 1 < opt['window_size']:
                            row += ' &_SENTENCE_END_& '
                    row += ' &_PARAGRAPH_END_& '

                row += '\n'
    
            sub_text_file.write(row)
        sub_text_file.close()
        '''

# vi:et:sw=4:ts=4
