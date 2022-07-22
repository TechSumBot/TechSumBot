# -*- coding: UTF-8 -*-

from nltk.stem.porter import *


def stemming(text):
    if len(text) < 2:
        return None
    stemmer = PorterStemmer()
    try:
        text = str(stemmer.stem(text))
    except Exception:
        return None
    return text


def stemming_for_word_list(word_list):
    new_word_list = []
    for word in word_list:
        if stemming(word) != None:
            new_word_list.append(stemming(word))
    return new_word_list


if __name__ == '__main__':
    print stemming("tests")
