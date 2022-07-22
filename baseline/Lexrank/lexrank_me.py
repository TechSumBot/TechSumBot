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
    resource = []
    with open(os.path.join(documents_dir,file),'rb') as f:
        str_file = str(file[2:-4])
        lxr = LexRank(str_file, stopwords=STOPWORDS['en'])
        new_dict = pickle.load(f)
        for order, item in enumerate(new_dict):
            SO_AnswerUnit_tmp = SO_Ans(item[0], item[3:], item[1], item[2])
            for sent in SO_AnswerUnit_tmp.body:
                resource.append(sent)
    summary = lxr.get_summary(resource, summary_size=5, threshold=0.1)
    with open('result/'+file[:-4]+'.txt','w') as f:
        for sent in summary:
            f.write(sent)





stop = timeit.default_timer()
print('Time: ', stop - start)
