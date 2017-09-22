from datetime import datetime

def field_array(data, field):
    return [d[field] for d in data]

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
