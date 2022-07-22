# -*- coding: utf-8 -*-
import os
from posixpath import split 
import pandas as pd
import re
import csv
from pandas.core.algorithms import mode
import nltk
import shutil
nltk.download('punkt')

def split_data(test_unit):

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(test_unit, 'html.parser')

    result = []
    output= []
    for s in soup('table'):
        s.string = '[table]'
        s.name = 'p'

    for s in soup('strong'):
        s.unwrap()

    for div in soup.find_all("div", {'class':'snippet'}): 
        div.string='[code snippet]'
        div.name = 'p'


    for s in soup('pre'):
        if s.code:
            s.code.unwrap()
        s.string = '[code snippet]'
        s.name = 'p'


    for s in soup('a'):
        hyper_link = s.get('href')
        # s.string='['+s.get_text()+']('+hyper_link+')'
        s.string='['+s.get_text()+' (hyper-link)]'
        s.unwrap()

    for s in soup('img'):
        s.string = '[image]'
        s.unwrap()

    for s in soup('li'):
        s.string = s.get_text()
        s.name = 'p'


    for p in soup('p'):
        result.append(p.get_text())

    for item in result:
        output+=nltk.sent_tokenize(item)

    return output

    