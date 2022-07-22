# -*- coding: UTF-8 -*-
import os
import sys

root_path = "/home/hywang/answerbot-tool/src"
sys.path.append(root_path)
import time
from utils.StopWords import remove_stopwords, read_EN_stopwords
from _1_question_retrieval.get_dq import get_dq, init_doc_matrix, init_doc_idf_vector
from _2_sentence_selection.get_ss import get_ss
from _3_summarization.get_summary import get_summary
from pathConfig import res_dir
from utils.csv_utils import write_list_to_txt
from utils.data_util import load_idf_vocab, load_w2v_model
from utils.data_util import preprocessing_for_query
from utils.db_util import read_all_questions_from_repo

def get_querylist(list_path):
    filr = open(list_path)
    query_list = []
    i = 0
    for line in filr:
        if i % 2 == 1:
            line = line.strip('\n')
            query_list.append(line)
        i += 1
    return query_list

def preprocess_all_questions(questions, idf, w2v, stopword):
    processed_questions = list()
    stopwords = stopword
    for question in questions:
        title_words = remove_stopwords(question.title, stopwords)
        if len(title_words) <= 2:
            continue
        if title_words[-1] == '?':
            title_words = title_words[:-1]
        question.title_words = title_words
        question.matrix = init_doc_matrix(question.title_words, w2v)
        question.idf_vector = init_doc_idf_vector(question.title_words, idf)
        processed_questions.append(question)
    return processed_questions


def main():

    for candidate in os.listdir("../../../dataset/input/json"):


        query=candidate.split('_')[1][:-4]
        topnum = 10
        file_path = '../../../dataset/input/json/'+candidate

        dq_res = list()
        stopword = read_EN_stopwords()


        top_dq_id_and_sim =[('testid',1)]
        # query = 'Why is list.size()>0 slower than list.isEmpty() in Java?'
        dq_res=[[query,top_dq_id_and_sim]]
        query_word = preprocessing_for_query(query)

        print 'sentence selection...', time.strftime('%Y-%m-%d %H:%M:%S')
        ss_res = list()
        for query, top_dq_id_and_sim in dq_res:
            top_ss = get_ss(query_word, topnum, top_dq_id_and_sim, stopword, file_path)
            ss_res.append((query, top_ss))
        # print ss_res
        print 'get summary...', time.strftime('%Y-%m-%d %H:%M:%S')
        print(candidate)
        res = list()
        try:
            for query, ss in ss_res:
                query = ' '.join(preprocessing_for_query(query))
                sum = get_summary(query, ss, 5)
                res.append([query, sum])

            res_fpath = os.path.join(res_dir, candidate[:-4]+'.txt')
            write_list_to_txt(res, res_fpath)
        except:
            print('\n\n\n\nThere is some fault\n\n\n\n')





if __name__ == '__main__':
    # test()
    main()