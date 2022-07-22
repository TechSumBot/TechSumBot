# -*- coding: UTF-8 -*-
import os
import sys
root_path = "/home/hywang/answerbot-tool/src"
sys.path.append(root_path)
from utils.db_util import read_correspond_answers_from_java_table
from utils.str_util import split_into_paragraph
import time
import operator
from _2_sentence_selection.Order.Order_Analysis import get_order_score
from _2_sentence_selection.Pattern.Pattern_Analysis import get_pattern_score
from _2_sentence_selection.HTMLTag.HTML_Analysis import get_html_score
from _2_sentence_selection.Entropy.Entropy_Analysis import get_entropy_score
from _2_sentence_selection.Entity.Entity_Analysis import get_entity_score, get_entities_from_word_list
import sys
from utils.preprocessing_util import preprocessing_for_ans_sent
from pathConfig import res_dir
import os
from utils.csv_utils import write_list_to_csv
from _2_sentence_selection.Entropy.build_tf_idf_dic import read_voc

def get_ss(query_word, top_relevant_paragraph_num, top_dq_id_and_sim, stopword, file_path):
    sent_list = []
    # Relevance
    max_Relevance = -sys.float_info.max
    min_Relevance = sys.float_info.max
    # Entity
    max_Entity = -sys.float_info.max
    min_Entity = sys.float_info.max
    # Score
    max_A_Score = -sys.float_info.max
    min_A_Score = sys.float_info.max
    # Order
    max_Order = -sys.float_info.max
    min_Order = sys.float_info.max
    # Pattern
    max_Pattern = -sys.float_info.max
    min_Pattern = sys.float_info.max
    # HTMLTag
    max_HTMLTag = -sys.float_info.max
    min_HTMLTag = sys.float_info.max
    # Entropy
    max_Entropy = -sys.float_info.max
    min_Entropy = sys.float_info.max
    # Score
    max_Score = -sys.float_info.max
    min_Score = sys.float_info.max

    # For Entropy calculation : preprocessing for query
    query_words = query_word
    # query entities
    query_entities = get_entities_from_word_list(query_words)
    # load stopwords
    stopwords = stopword
    # load idf voc
    idf_voc = read_voc()
    # question-level

    answers_list = read_correspond_answers_from_java_table(top_dq_id_and_sim,file_path)
    '''
    answers list is a list for relevant answers
    each row [answer id, votes, question id, *answer sentences]
    '''
    for (q_id, sim) in top_dq_id_and_sim:
        #answers = read_correspond_answer_from_java_table(q_id)
        # answer-level
        answers = answers_list[q_id]
        print(q_id)
        print('q_id')

        for answer_tmp in answers:
            # print(answer_tmp.__dict__)
            answer_body = answer_tmp.body
            # sentences = split_into_paragraph(answer_body)
            sentences = answer_body
            order = 1
            a_id = answer_tmp.id
            # paragraph-level
            for sent in sentences:
                clean_sent = preprocessing_for_ans_sent(sent)
                # relevance
                relevance = sim
                max_Relevance = relevance if max_Relevance < relevance else max_Relevance
                min_Relevance = relevance if min_Relevance > relevance else min_Relevance
                # entity
                entity = get_entity_score(query_entities, clean_sent)
                max_Entity = entity if max_Entity < entity else max_Entity
                min_Entity = entity if min_Entity > entity else min_Entity
                # score
                a_score = answer_tmp.score
                max_A_Score = a_score if max_A_Score < a_score else max_A_Score
                min_A_Score = a_score if min_A_Score > a_score else min_A_Score
                # Order
                order = get_order_score(order)
                order += 1
                max_Order = order if max_Order < order else max_Order
                min_Order = order if min_Order > order else min_Order
                # Pattern
                pattern = get_pattern_score(clean_sent)
                max_Pattern = pattern if max_Pattern < pattern else max_Pattern
                min_Pattern = pattern if min_Pattern > pattern else min_Pattern
                # HTML Tag
                htmltag = get_html_score(sent)
                max_HTMLTag = htmltag if max_HTMLTag < htmltag else max_HTMLTag
                min_HTMLTag = htmltag if min_HTMLTag > htmltag else min_HTMLTag
                # Entropy
                entropy = get_entropy_score(query_words, clean_sent, stopwords, idf_voc)
                max_Entropy = entropy if max_Entropy < entropy else max_Entropy
                min_Entropy = entropy if min_Entropy > entropy else min_Entropy
                sent_list.append([clean_sent, relevance, entity, a_score, order, pattern, htmltag, entropy, a_id, q_id])


    # Normalization from 1.0 -> 2.0 except Score
    Normalized_sent_list_tmp = []
    for [clean_sent, relevance, entity, a_score, order, pattern, htmltag, entropy, a_id, q_id] in sent_list:
        relevance = 1.0 if (max_Relevance - min_Relevance) == 0 else 1.0 + (relevance - min_Relevance) / (
                max_Relevance - min_Relevance)
        entity = 1.0 if (max_Entity - min_Entity) == 0 else 1.0 + (entity - min_Entity) / (max_Entity - min_Entity)
        a_score = 1.0 if (max_A_Score - min_A_Score) == 0 else 1.0 + (a_score - min_A_Score) / (
                max_A_Score - min_A_Score)
        order = 1.0 if (max_Order - min_Order) == 0 else 1.0 + (order - min_Order) / (max_Order - min_Order)
        pattern = 1.0 if (max_Pattern - min_Pattern) == 0 else 1.0 + (pattern - min_Pattern) / (
                max_Pattern - min_Pattern)
        htmltag = 1.0 if (max_HTMLTag - min_HTMLTag) == 0 else 1.0 + (htmltag - min_HTMLTag) / (
                max_HTMLTag - min_HTMLTag)
        entropy = 1.0 if (max_Entropy - min_Entropy) == 0 else 1.0 + (entropy - min_Entropy) / (
                max_Entropy - min_Entropy)
        Score = relevance * entity * a_score * order * pattern * htmltag * entropy
        max_Score = Score if max_Score < Score else max_Score
        min_Score = Score if min_Score > Score else min_Score
        Normalized_sent_list_tmp.append([clean_sent, Score, a_id, q_id])
    del sent_list

    # Normalization Score from 0.0 -> 1.0
    Normalized_sent_list = []
    for [clean_sent, Score, a_id, q_id] in Normalized_sent_list_tmp:
        Score = (Score - min_Score) / (max_Score - min_Score)
        Normalized_sent_list.append([clean_sent, Score, a_id, q_id])
    del Normalized_sent_list_tmp

    # sort by Score then q_id
    Normalized_sent_list.sort(key=operator.itemgetter(1, 3), reverse=True)
    return [x[0] for x in Normalized_sent_list[:top_relevant_paragraph_num]]


def load_qs_result(rq_fpath):
    import pandas as pd
    rq_res = []
    df = pd.read_csv(rq_fpath)
    for idx, row in df.iterrows():
        rq_res.append([row[0], eval(row[1])])
    return rq_res


if __name__ == '__main__':
    print 'start : ', time.strftime('%Y-%m-%d %H:%M:%S')
    # parameter
    top_relevant_paragraph_num = 10
    rq_res_fpath = os.path.join(res_dir, "rq_res.csv")
    res = []
    for query, top_dq_id_and_sim in load_qs_result(rq_res_fpath):
        top_ss = get_ss(query, top_relevant_paragraph_num, top_dq_id_and_sim)
        for i in range(top_relevant_paragraph_num):
            print("#%s\nsent: %s\n\n" % (i, top_ss[i]))
        res.append([query, top_ss])


    res_fpath = os.path.join(res_dir, 'ss_res.csv')
    header = ["query", "ss"]
    write_list_to_csv(res, res_fpath, header)

    print 'Done. ', time.strftime('%Y-%m-%d %H:%M:%S')
