import timeit
from lexrank import STOPWORDS, LexRank
import spacy
from spacy.lang.en import English
import pickle
import os
from SO_ans import SO_Ans
start = timeit.default_timer()
documents = []
documents_dir = '../../dataset/input/json'


for file in os.listdir(documents_dir): 
  print(file)
  resource = ''
  with open(os.path.join(documents_dir,file),'rb') as f:
    str_file = str(file[2:-4])
    new_dict = pickle.load(f)
    for order, item in enumerate(new_dict):
      SO_AnswerUnit_tmp = SO_Ans(item[0], item[3:], item[1], item[2])
      with open('data/'+file[:-4]+'.txt','a') as f:
        for sent in SO_AnswerUnit_tmp.body:
          f.write(sent)
          f.write('\n')
  
    cmd = 'sumy lex-rank --length=5 --file=\'data/'+file[:-4]+'.txt\''
    print(cmd)
    res = os.popen(cmd)
    output_str = res.read()   
    print(output_str)