import os
import re
from tkinter import N
from tkinter.messagebox import QUESTION
import shutil


def get_key(val,my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key
 
    return "There is no such Key"

def get_query_from_summary():
    '''
    get the queries name of the gold summaries as a dic
    '''
    dir = 'gold'
    output = {}
    for file in os.listdir(dir):
        sent_rule = '(?<=_)[A-Z].*\.txt'
        num_rule = '(?<=[0-9]_)[0-9]*'
        sent = file
        searchobj1 = re.search(sent_rule,sent)
        searchobj2 = re.search(num_rule,sent)

        # print(searchobj2[0])
        output[searchobj2[0]]=searchobj1[0][:-4]


    return output


def rename_input(query_dic):
    count = 0
    for dir in os.listdir('input'):
        for file in os.listdir(os.path.join('input',dir)):
            if file.startswith('Guid')==False and file.startswith('.DS')==False :
                # print(file)
                with open(os.path.join('input',dir,file),'r')as f:
                    next(f)
                    sent = f.readline()
                    query = sent[9:-2]
                    num = get_key(query,query_dic)
                    # print(num)
                    file_name = num+'_'+query+'.txt'
                try:
                    shutil.copy(os.path.join('input',dir,file), os.path.join('input','overall',file_name))
                except IOError as e:
                    print("Unable to copy file. %s" % e)

def delete_invalid_charactor(file_path,new_path):
    '''
    replace the invalid charactor for rouge score (e.g., <> as \(\))
    input: file_path
    '''
    f2 = open(new_path, 'r+')
    with open(file_path,'r+') as f1:
        str1 = '<'
        rep1 = '>'
        for sent in f1.readlines():
            tt = re.sub(str1,rep1,sent)
            f2.write(tt)
    f2.close()


def main():
    return


if __name__ =='__main__':
    main()
    # query_list = get_query_from_summary()
    # rename_input(query_list)

