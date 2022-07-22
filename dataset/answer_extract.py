import os
import re
import util
import midresult.query_summary_preprocess.sql_function as sql
dir = 'input/overall'
import json
import pickle


for file in os.listdir(dir):

    print(file)

    outrule = '[0-9]*(?=_)'
    query_num = re.search(outrule,file)
    # print(query_num[0])
    output = []

    flag = -1

    with open(os.path.join(dir,file)) as f:
        # print(f.readlines()[6])
        for sent in f.readlines()[6:]:
            answer_rule = '(?<=#)[0-9]*(?=\s\(http)'
            answerid_rule = '(?<=\/)[0-9]*(?=\))'
            answer_num = re.search(answer_rule,sent)
            answerid = re.search(answerid_rule,sent)
            
            if answer_num:
                if int(answer_num[0])!=0:      
                    output.append(answer)
                answer = []

                query_id = sql.get_parientid(answerid[0])[0]['ParentId']
                votes = sql.get_votes(answerid[0])[0]['Score']
                flag = answer_num[0]
                # print(flag)
                # answer[0]=query_id
                answer.append(answerid[0])
                answer.append(votes)
                answer.append(query_id)
            else:
                if sent.startswith(' ['):
                    answer.append(sent[10:-2])
    # print(output)
    with open('input/json/'+file[:-4]+'.pkl','wb') as f:
        pickle.dump(output,f,protocol=2)
