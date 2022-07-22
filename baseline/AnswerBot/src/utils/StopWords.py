# -*- coding: UTF-8 -*-
from pathConfig import get_base_path
from nltk import word_tokenize

path_of_stopwords_EN = get_base_path() + '/utils/StopWords_EN.txt'


def read_EN_stopwords():
    sw_set = set()
    f = open(path_of_stopwords_EN)
    for line in f:
        sw_set.add(line.strip())
    return sw_set


def remove_stopwords(sent, sw):
    if type(sent) is str:
        wlist = word_tokenize(sent)
    elif type(sent) is list:
        wlist = sent
    else:
        raise Exception("Wrong type for removing stopwords!")
    sent_words = []
    for w in wlist:
        if w == '':
            continue
        if w not in sw:
            sent_words.append(w)
    return sent_words
