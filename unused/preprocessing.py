# coding=utf8
import re
import pandas as pd
import numpy as np
import json
import nltk

from nltk.corpus import stopwords
from variation import Variation

def remove_year(vText):
    """ Remove Text with years like: (1999)
    Args:
      text: str
    Returns:
      without year String.
    """
    return re.sub(r"(\d{4})", "", vText) #maybe reduction
def remove_citation(vText):
    """ Remove Text with citation. [3], [10-12], [3, 4, 6], (1),  extra remove:(22%, n = 8/37)
    Args:
      text: str
    Returns:
      without citation String.
    """
    vText = re.sub(r"\([^\)]*,[^\)]*\)", "", vText) # (DMN; Raichle et al., 2001;)
    vText = re.sub(r"\[\d\{1,2}\]","" , vText) #[1]
    vText = re.sub(r"\[\d{1,2}\s*-\s*\d{1,2}\]","" , vText)# [10-12]
    vText = re.sub(r"\[[0-9]+(\ *,\ *[0-9]+\ *)*\]","" , vText) #[1,3,4]
    vText = re.sub(r"\([0-9]+(\ *,\ *[0-9]+\ *)*\)","" , vText) #(1,3,4)
    vText = re.sub(r"\(\d*\)","" , vText)#(3)
    #vText = re.sub(r"\[\d\{1,2}\{,\d\{1,2}}]+","" , vText)
    return vText

def remove_http(vText):
    """ Remove specific string.
    Args:
      text: str
    Return:
      clear String
    """
    vText = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vText, flags=re.MULTILINE) #remove url
    vText = re.sub(r"(e-?)?mail: ([\w+-]+[\w.+-]*@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)", '', vText, flags=re.IGNORECASE) #remove email
    vText = re.sub(r"([\w+-]+[\w.+-]*@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)", '', vText, flags=re.MULTILINE) #remove e-mail
    vText = re.sub(r"[\(](supplementary|fig)\.?.*[\)]\.?", "" , vText,flags=re.IGNORECASE) #remove (fig) or (supplementary)
    return vText

def remove_stopwords(vText):
    """ Remove Stopwords in NLTK Stopwords List
    Args:
      text: str
    Returns:
        clear string
    """
    stopwords_list = stopwords.words('english')
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s+')
    vText = pattern.sub("", vText)
    return vText

varalias = json.load(open("one2many.json"))
# Read input file
train_text = pd.read_csv("input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text = pd.read_csv("input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
train = pd.read_csv('input/training_variants')
test = pd.read_csv('input/test_variants')


def preprocessing(text, gene, var):
    """ replace many amino to 1 amino.
        Returns:
          replace_text.
    """
    var = Variation(var)
    text = remove_year(text)
    text = remove_citation(text)
    text = remove_http(text)
    text = remove_stopwords(text)
    # Handling Variation
    if var.type == "point": #re format: "^([A-Za-z])(\d+)([A-Za-z\*])", including *
        if var.end_amino == "*":
            alias_list = []+["%s%s\S*" % (start_m, var.pos) for start_m in [var.start_amino]+varalias[var.start_amino.upper()]]
        elif var.end_amino == "":
            alias_list = ["%s%s" % (start_m, var.pos) for start_m in varalias[var.start_amino.upper()]]
        else:
            alias_list = ["%s%s%s" % (start_m, var.pos, end_m) for start_m in varalias[var.start_amino.upper()] for end_m in varalias[var.end_amino.upper()] ]
        # replace many to 1
        text = re.sub("%s" % "|".join(alias_list), var.var, text, flags = re.IGNORECASE)
    return text
out_f = open("output/train","w")
out_f.write("ID,Text\n")
for i in range(len(train)):
    text = train_text.Text[i]
    gene = train.Variation[i]
    var = train.Variation[i]
    text = preprocessing(text, gene, var)
    out_f.write(str(i)+"||"+ text+"\n")
out_f.close()

out_f = open("output/test","w")
out_f.write("ID,Text\n")
for i in range(len(test)):
    text = test_text.Text[i]
    gene = test.Variation[i]
    var = test.Variation[i]
    text = preprocessing(text, gene, var)
    out_f.write(str(i)+"||"+ text+"\n")
out_f.close()
