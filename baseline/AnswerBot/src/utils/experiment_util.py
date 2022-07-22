# -*- coding: UTF-8 -*-

from utils.file_util import write_pdf_file, write_file
from pathConfig import get_base_path


# format : [id, title, final_similarity]


def save_dq_result_baseline(queryNum, query, top_dq, path):
    pdf_str_tmp = 'Q_No.' + str(queryNum) + '\n'
    pdf_str_tmp += ('Query : ' + query + '\n\n')
    txt_str_tmp = (query + '\n')
    Num = 1
    for [id, title, final_similarity] in top_dq:
        # pdf str
        pdf_str_tmp += ('No.' + str(Num) + '\n')
        pdf_str_tmp += ('Title : ' + str(title) + '\n')
        pdf_str_tmp += ('Link : http://stackoverflow.com/questions/' + str(id) + '\n')
        pdf_str_tmp += '\n'
        Num += 1
        # txt str
        txt_str_tmp += (str(id) + '\n')
        txt_str_tmp += (str(title) + '\n')
        txt_str_tmp += (str(final_similarity) + '\n')
    write_pdf_file(path + '.pdf', pdf_str_tmp.strip().split('\n'))
    write_file(path + '.txt', txt_str_tmp.strip())


# format : [id, entities, title, final_similarity]

def save_dq_result_our_approach(query, entities, top_dq, path, high_relevant_id, rank):
    str_tmp = query + '\n'
    for entity in entities:
        str_tmp += (str(entity) + ' ')
    str_tmp = str_tmp.strip() + '\n'
    for [id, title, final_similarity] in top_dq:
        str_tmp += (str(id) + '\n')
        str_tmp += (str(title) + '\n')
        str_tmp += (str(final_similarity) + '\n')
    for id in high_relevant_id:
        str_tmp += (str(id) + ' ')
    str_tmp = str_tmp.strip() + '\n'
    for i in range(0, 100, 1):
        str_tmp += (str(rank[i]) + ' ')
    write_file(path, str_tmp.strip())
    # print 'save done.'


# format : [id, title, final_similarity]

def load_dq_result_baseline(path):
    top_dq = []
    file = open(path)
    count = 0
    for line in file:
        line = line.strip()
        if count == 0:
            query = line
        elif count % 3 == 1:
            id = int(line)
        elif count % 3 == 2:
            title = line
        else:
            sim = float(line)
            top_dq.append([id, title, sim])
        count += 1
    return query, top_dq


# format : [id, entities, title, final_similarity]

def load_dq_result_our_approach(path):
    top_dq = []
    file = open(path)
    count = 0
    for line in file:
        line = line.strip()
        # first line : query
        if count == 0:
            query = line
        # second line : entities
        elif count == 1:
            entities = line.split(' ')
        elif count % 3 == 2:
            id = int(line)
        elif count % 3 == 0:
            title = line
        else:
            sim = float(line)
            top_dq.append([id, title, sim])
        count += 1
    return query, entities, top_dq


# top_ss format : [sent_Num, raw_sent, sent_without_tag, Order, Score, q_id]

def save_ss_result_our_approach(query, top_ss, summary, path):
    str_tmp = query + '\n'
    # str_tmp += ('Relevant Sentences : \n')
    # for [sent, Relevance, Entity, A_Score, Order, Pattern, HTMLTag, Entropy, Score, q_id] in top_ss:
    #     str_tmp += (str(sent) + '\n')
    #     str_tmp += (str(Score) + '\n')
    str_tmp += ('Summary : \n' + summary)
    write_file(path, str_tmp)


def load_id_list_from_ss_result(dir):
    id_list = []
    for i in range(0, 100, 1):
        path = dir + '/' + str(i) + '.txt'
        file = open(path)
        id_list_tmp = []
        linenum = 0
        for line in file:
            line = line.strip()
            if linenum != 0 and linenum != 1:
                id_list_tmp.append(line.split(' $$ ')[0])
            linenum += 1
        id_list.append(id_list_tmp)
    return id_list


# [query, top_dq]
# top dq : [id, text, relevance]

def load_Step1_result(approach_name, topnum):
    dir_of_result = get_base_path() + '/_1_Result/Baseline_' + approach_name
    result = []
    for i in range(0, 100, 1):
        path_of_result = dir_of_result + '/' + str(i) + '.txt'
        file = open(path_of_result)
        linenum = 0
        top_dq = []
        for line in file:
            line = line.strip()
            if linenum == 0:
                query = line
            elif linenum % 3 == 1:
                id = line
            elif linenum % 3 == 2:
                title = line
            else:
                sim = float(line)
                top_dq.append([id, title, sim])
                if len(top_dq) >= topnum:
                    break
            linenum += 1
        result.append([query, top_dq])
    return result
