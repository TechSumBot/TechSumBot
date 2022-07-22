import os
import util.sql as sql
import pickle
import re
from random import choice
import csv
import pandas as pd
########################
# to build the triplet data for contrastive learning training
# data_1: original query
# data_2: positive query
# data_3: negative query
########################


###############
## for java
# pairs = sql.get_duplicate_pairs('java')
# with open ('data/pairs_java.pkl','wb') as f:
#     pickle.dump(pairs,f) 
# get all java answers
# all_candidate = sql.get_all_java_answers()
# with open ('data/candidate_all_java_questions.pkl','wb') as f:
    # pickle.dump(all_candidate,f)
################    
## for python
# pairs = sql.get_duplicate_pairs('python')
# with open ('data/pairs_python.pkl','wb') as f:
#     pickle.dump(pairs,f) 
# get all python answers
# all_candidate = sql.get_all_python_answers()
# with open ('data/candidate_all_python_questions.pkl','wb') as f:
#     pickle.dump(all_candidate,f)
def identify_same_tags(tag_1, tag_2, tag_3, target_tag):
    #identify whether the third tag list have at least one same tag with the first or second tag list (except target tag).
    #output: true/false
    # test1 = ['python','php','go']
    # test2 = ['nodejs','javascript','go','java']
    # test3 = ['go','java']
    list_tmp_1 = list(set(tag_1) & set(tag_3))
    if list_tmp_1 != [target_tag]:
        return True
    list_tmp_2 = list(set(tag_2) & set(tag_3))
    if list_tmp_2 != [target_tag]:
        return True
    return False       

def main(pl):
    with open ('data/candidate_all_'+pl+'_questions.pkl','rb') as f:
        candidate=pickle.load(f) #读取文件到list
    with open ('data/pairs_'+pl+'.pkl','rb') as f:
        pairs=pickle.load(f) #读取文件到list
    print(candidate[0])

    output = []

    for pair in pairs:

        query_1 =sql.get_query_name(str(pair['RelatedPostId']))
        query_2 =sql.get_query_name(str(pair['postid']))   
        tag_1 = sql.get_post_tags(str(pair['RelatedPostId']))
        tag_2 =sql.get_post_tags(str(pair['postid']))
        if not tag_1 or (not tag_2):
            continue
        tag_1 = tag_1[0]['tags']
        tag_2 = tag_2[0]['tags']
        tag_rule = '(?<=<)[a-zA-Z\-]*?(?=>)'
        tag_1 = re.findall(tag_rule,tag_1)
        tag_2 = re.findall(tag_rule,tag_2)
        if str(pl) not in tag_1 or str(pl) not in tag_2:
            continue
        query_1 = query_1[0]['Title']
        query_2 = query_2[0]['Title']
        # randomly select one query
        random_flag = True

        if not query_1 or (not query_2):
            continue

        while(random_flag):
            query_3 = choice(candidate)
            # print(query_3)
            # print(tag_1)
            # print(query_1)

            # print(tag_2)
            # print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            tag_3 = sql.get_post_tags(str(query_3['id']))
            tag_3 = re.findall(tag_rule,tag_3[0]['tags'])
        # identify if the randonly selected query 
        # don't share none of the tags with above two queries 
        # except the java tag
            if not identify_same_tags(tag_1,tag_2,tag_3,str(pl)):
                random_flag = False
        query_3 = query_3['title']
        # print(query_1)
        # print(query_2)
        # print(query_3)
        output.append([query_1, query_2, query_3])
    with open('data/train_'+str(pl)+'.tsv','w')as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(output) 
def combine_training_file():
    all = ['data/train_python.tsv','data/train_java.tsv']
    df_from_each_file = (pd.read_csv(f, sep='\t') for f in all)
    for file in df_from_each_file:
        file.to_csv( "data/train.csv",index=False, mode='a+')
        print(file)




if __name__ == "__main__":
    # main('java')
    # main('python')
    combine_training_file()
