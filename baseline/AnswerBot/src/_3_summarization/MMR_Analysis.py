# -*- coding: UTF-8 -*-

from utils.StopWords import read_EN_stopwords
import sys
import copy
from utils.data_util import load_idf_vocab, load_w2v_model
from utils.StopWords import remove_stopwords
from nltk import word_tokenize
from _1_question_retrieval.get_dq import calc_similarity


# top_ss format: [sent_Num, raw_sent, sent_without_tag, Order, Score, q_id]

def MMR_Analysis(query, top_ss, topnum):
    # print 'MMR analysis', time.strftime('%Y-%m-%d %H:%M:%S')
    sim_matrix = build_sim_matrix(query, top_ss)
    rank_list = MMR_Algorithm(sim_matrix, topnum)
    selected_sentence = []
    for rank in rank_list:
        selected_sentence.append(top_ss[rank])
    return selected_sentence


######### Matrix #########
#      d1 d2 ... dn query
# d1   1  S1,2  S1,n
# d2      1
# ...        ...
# dn             1
# query              1
##########################

'''
top_ss format : [sent_Num, raw_sent, sent_without_tag, Order, Score, q_id]
'''


def build_sim_matrix(query, top_ss):
    # add query
    top_ss_tmp = copy.deepcopy(top_ss)
    top_ss_tmp.append(query)
    len_of_paragraph = len(top_ss_tmp)
    sim_matrix = [[0 for col in range(len_of_paragraph)] for row in range(len_of_paragraph)]
    # doc sim parameter
    w2v_model = load_w2v_model()
    stopwords = read_EN_stopwords()
    idf_voc = load_idf_vocab()

    # tokenize
    for i in range(len(top_ss_tmp)):
        top_ss_tmp[i] = word_tokenize(top_ss_tmp[i])
        top_ss_tmp[i] = remove_stopwords(top_ss_tmp[i], stopwords)

    for i in range(0, len_of_paragraph, 1):
        for j in range(0, i + 1, 1):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                sim_matrix[i][j] = (calc_similarity(top_ss_tmp[i], top_ss_tmp[j], idf_voc, w2v_model) +
                                    calc_similarity(top_ss_tmp[j], top_ss_tmp[i], idf_voc, w2v_model)) / 2.0
                sim_matrix[j][i] = sim_matrix[i][j]
    return sim_matrix


def MMR_Algorithm(sim_matrix, topnum):
    iteration = 1
    query_index = len(sim_matrix) - 1
    # init
    Set = []
    Rest = [i for i in range(0, query_index - 1, 1)]
    # parameter
    Lambda = 0.5
    while iteration <= topnum:
        # find most sim with query
        most_sim_with_query_index = -1
        max_dq_sim = -1
        for i in range(0, query_index, 1):
            if sim_matrix[i][query_index] > max_dq_sim:
                max_dq_sim = sim_matrix[i][query_index]
                most_sim_with_query_index = i
        if len(Set) == 0:
            Set.append(most_sim_with_query_index)
            try:
                Rest.remove(most_sim_with_query_index)
            except:
                print('error')
        else:
            max_MMR = -sys.float_info.max
            max_MMR_idx = -1
            for cur in Rest:
                max_dd_sim = -sys.float_info.max
                max_dd_idx = -1
                for i in Set:
                    if sim_matrix[cur][i] > max_dd_sim:
                        max_dd_sim = sim_matrix[cur][i]
                        max_dd_idx = i
                MRR_tmp = Lambda * sim_matrix[cur][query_index] - (1 - Lambda) * sim_matrix[cur][max_dd_idx]
                if MRR_tmp > max_MMR:
                    max_MMR = MRR_tmp
                    max_MMR_idx = cur
            Set.append(max_MMR_idx)
            Rest.remove(max_MMR_idx)
        iteration += 1
    return Set


######## Test Matrix #########
# [1,0.11,0.23,0.76,0.25,0.91]
# [0.11,1,0.29,0.57,0.51,0.90]
# [0.23,0.29,1,0.02,0.20,0.50]
# [0.76,0.57,0.02,1,0.33,0.06]
# [0.25,0.51,0.20,0.33,1,0.63]
# [0.91,0.90,0.50,0.06,0.63,1]
##############################


if __name__ == '__main__':
    sim_matrix = [[1, 0.11, 0.23, 0.76, 0.25, 0.91], [0.11, 1, 0.29, 0.57, 0.51, 0.90],
                  [0.23, 0.29, 1, 0.02, 0.20, 0.50], [0.76, 0.57, 0.02, 1, 0.33, 0.06],
                  [0.25, 0.51, 0.20, 0.33, 1, 0.63], [0.91, 0.90, 0.50, 0.06, 0.63, 1]]
    topnum = 3
    Set = MMR_Algorithm(sim_matrix, topnum)
    print Set
