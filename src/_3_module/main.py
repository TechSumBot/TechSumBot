import summa.summarizer as summarizer
import os
import pickle

documents_dir = '../../../dataset/input/json'

class SO_Ans:
    __slots__ = 'id', 'body', 'score', 'parent_id'

    def __init__(self, id, body, score, parent_id):
        self.id = id
        self.body = body
        self.score = score
        self.parent_id = parent_id


for file in os.listdir(documents_dir): 
    resource = []
    with open(os.path.join(documents_dir,file),'rb') as f:
        new_dict = pickle.load(f)
        for order, item in enumerate(new_dict):
            SO_AnswerUnit_tmp = SO_Ans(item[0], item[3:], item[1], item[2])
            for sent in SO_AnswerUnit_tmp.body:
                resource.append(sent)
    
    summary = summarizer.summarize(resource, sent_length=5)
    with open('result/'+file[:-4]+'.txt','w') as f:
        for sent in summary:
            f.write(sent)