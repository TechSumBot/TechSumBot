# -*- coding: utf-8 -*-
import os
import sys
root_path = "/home/hywang/answerbot-tool/src"
sys.path.append(root_path)
import operator
import time
from utils.StopWords import remove_stopwords, read_EN_stopwords
from utils.db_util import read_all_questions_from_repo, read_specific_question_from_repo
import numpy as np

def init_doc_matrix(doc,w2v):

    matrix = np.zeros((len(doc),200)) #word embedding size is 100
    for i, word in enumerate(doc):
        if word in w2v.wv.vocab:
            matrix[i] = np.array(w2v.wv[word])

    #l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
        #matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
    except RuntimeWarning:
        print doc

    #matrix = np.array(preprocessing.normalize(matrix, norm='l2'))

    return matrix

def init_doc_idf_vector(doc,idf):
    idf_vector = np.zeros((1,len(doc)))  # word embedding size is 200
    for i, word in enumerate(doc):
        if word in idf:
            idf_vector[0][i] = idf[word]

    return idf_vector


def sim_doc_pair(matrix1,matrix2,idf1,idf2):

    sim12 = (idf1*(matrix1.dot(matrix2.T).max(axis=1))).sum() / idf1.sum()

    sim21 = (idf2*(matrix2.dot(matrix1.T).max(axis=1))).sum() / idf2.sum()

    return (sim12 + sim21) / 2.0
    # total_len = matrix1.shape[0] + matrix2.shape[0]
    # return sim12 * matrix2.shape[0] / total_len + sim21 * matrix1.shape[0] / total_len

def calc_wordvec_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, len(vec1))
    vec2 = vec2.reshape(1, len(vec2))
    x1_norm = np.sqrt(np.sum(vec1 ** 2, axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(vec2 ** 2, axis=1, keepdims=True))
    prod = np.sum(vec1 * vec2, axis=1, keepdims=True)
    cosine_sim = prod / (x1_norm * x2_norm)
    return cosine_sim[0][0]


def calc_similarity(word_list_1, word_list_2, idf_voc, word2vector_model):
    if len(word_list_1) == 0 or len(word_list_2) == 0:
        return 0.0

    sim_up = 0
    sim_down = 0
    for w1 in word_list_1:
        w1_unicode = w1.decode('utf-8')
        if w1_unicode in word2vector_model:
            w1_vec = word2vector_model[w1_unicode]
            # maxsim
            maxsim = 0.0
            for w2 in word_list_2:
                # word similarity
                w2_unicode = w2.decode('utf-8')
                if w2_unicode in word2vector_model:
                    w2_vec = word2vector_model[w2_unicode]
                    sim_tmp = calc_wordvec_similarity(w1_vec, w2_vec)
                    if sim_tmp > maxsim:
                        maxsim = sim_tmp
            # if exist in idf
            if w1 in idf_voc:
                idf = idf_voc[w1]
                sim_up += maxsim * idf
                sim_down += idf
            # else:
            # print("%s not in idf vocabulary!" % w1)
    if sim_down == 0:
        print("sim_down = 0!\n word sent 1 %s\nword sent 2 %s" % (word_list_1, word_list_2))
        return 0
    return sim_up / sim_down


def get_dq(query_w, topnum, questions, query_idf, query_matrix):
    rank = []
    #stopwords = read_EN_stopwords()
    cnt = 0
    for question in questions:
        #title_w = remove_stopwords(q.title, stopwords)
        #sim = calc_similarity(query_w, repo_idtitle[key], query_idf, query_matrix)
        #sim += calc_similarity(repo_idtitle[key], query_w, query_idf, query_matrix)
        # sim /= 2.0

        sim = sim_doc_pair(query_matrix,question.matrix, query_idf, question.idf_vector)
        rank.append([question.id, sim])
        cnt += 1
        if cnt % 10000 == 0:
            print("Processed %s questions...%s" % (cnt, time.strftime('%Y-%m-%d %H:%M:%S')))

    # format: [id,sim]
    rank.sort(key=operator.itemgetter(1), reverse=True)
    # top_dq,rank
    top_dq = []
    for i in range(0, len(rank), 1):
        id = rank[i][0]
        sim = rank[i][1]
        rank.append(id)
        if i < topnum:
            qs = read_specific_question_from_repo(id)
            top_dq.append((qs, sim))
    return top_dq

