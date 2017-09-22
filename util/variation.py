import re

import numpy as np
import pandas as pd


class Variation(object):
    def __init__(self, var):
        self.var = var
        self.type = ''
        self.start_pos = 0
        self.end_pos = 0
        self.start_amino = ''
        self.end_amino = ''
        self.pos = 0
        self.stop_sign = False
        
        self.alias_dict = {'del': 'deletion', 'ins': 'insertion',
                           'dup': 'duplication'}

        self.parse()

    def to_dict(self):
        return {'type': self.type, 'start_pos': self.start_pos,
                'end_pos': self.end_pos, 'start_amino': self.start_amino,
                'end_amino': self.end_amino, 'pos': self.pos,
                'stop_sign': self.stop_sign} 

    def parse(self):
        """This is a naive parser
        using a lot of heuristics and observations
        """

        special_vars = {'amplification', 'copy number loss', 
            'epigenetic silencing', 'overexpression'}

        special_terms = ['dna binding domain', 'egfrv', 'truncating mutation',
                        'fusion', 'mutation', 'deletion', 'duplication', 'insertion',
                        'hypermethylation']

        var = self.var.lower()

        # Check if the stop sign '*' in the variation
        if '*' in var:
            self.stop_sign = True
        
        # Type "exact match with special pre-difined variations"
        if var in special_vars:
            self.type = var
            return
        
        # Type "with special term"
        for term in special_terms:
            if term in var:
                self.type = term
                return

        # Type "point": A123B or A123* or A123
        if re.match('^[a-z][0-9]+[a-z|*]?$', var):
            split = re.split('[0-9]+', var)
            self.type = 'point'
            self.start_amino = split[0]
            self.end_amino = split[1]
            s = re.search('[0-9]+', var)
            self.pos = int(s.group())
            return

        # Type "del/ins/trunc/splice/dup/fs": A123del or A123_B234del
        for suffix in ['del', 'ins', 'trunc', 'splice', 'dup', 'fs']:
            if suffix in var:
                self.type = self.alias_dict.get(suffix, suffix)
                self._parse_suffix(var, suffix)
                return

        print('[INFO] variation cannot be parsed: %s' % self.var)

    def _parse_suffix(self, var, suffix):
        var_nosuffix = var.split(suffix)[0]
        if re.match('^[a-z]?[0-9]+[a-z]?[_]?$', var_nosuffix):
            # ex: T123del or T123_splice
            var_nosuffix = var_nosuffix.replace('_', '')
            self.start_amino, self.pos = self._parse_amino_pos(var_nosuffix)
        elif re.match('^[a-z]?[0-9]+[a-z]?_[a-z]?[0-9]+[a-z]?$', var_nosuffix):
            # ex: T123_A345del
            start, end = re.split('_', var_nosuffix)
            self.start_amino, self.start_pos = self._parse_amino_pos(start)
            self.end_amino, self.end_pos = self._parse_amino_pos(end)
        return

    def _parse_amino_pos(self, var):
        """This function is for parsing amino and pos
        from "t123" or "123" or "t123r"(last r is ignored)
        """
        if re.match('[a-z]', var[-1]):
            var = var[:-1]

        amino = ''
        if re.match('^[a-z]', var):
            amino = var[0]
            pos = int(var[1:])
        else:
            pos = int(var)

        return amino, pos


class Variations:
    def __init__(self, var_list):
        self.list = list(var_list)
        print('[INFO] There are %d variations in the list' % len(self.list))
        self.var_list = []
        for var in self.list:
            Var = Variation(var)
            self.var_list.append(Var)
    
    def to_df(self):
        var_dicts = [Var.to_dict() for Var in self.var_list]
        df = pd.DataFrame(var_dicts)
        return df

    def to_dummy_df(self):
        dummy_columns = ['start_amino', 'end_amino', 'type']
        df = self.to_df()
        df_dummy = pd.get_dummies(df, columns=dummy_columns)
        return df_dummy

