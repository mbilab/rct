from datetime import datetime
from os.path import isfile
from pickle import dump

def field_array(data, field):
    return [d[field] for d in data]

def save(data, filename):
    if isfile(filename):
        raise FileExistsError('%s existed, please delete it manually before saving' % (filename))
    dump(data, open(filename, 'wb'))
    print('%s saved' % (filename))

def tick(last=None):
    n = datetime.now()
    if last:
        print(n - last)
    return n

def to_csv(data, fields):
    print(','.join(fields))
    for d in data:
        print(','.join([d[f] for f in fields]))

# vi:et:sw=4:ts=4
