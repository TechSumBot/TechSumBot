# -*- coding: UTF-8 -*-

# import pymysql as mdb
from data_structure.SO_que import SO_Que
from data_structure.SO_ans import SO_Ans
from preprocessing_util import preprocessing_for_que, preprocessing_for_ans
from utils.time_utils import get_current_time
import numpy as np
import pickle
# repo
def read_questions_from_repo(num):
    sql = 'SELECT * FROM answerbot.repo WHERE PostTypeId=1 limit 0,' + str(num)
    SO_datalist = []
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    try:
        cur.execute(sql)
        results = cur.fetchall()
        count = 0
        for row in results:
            count += 1
            # id,type,title,title_NO_SW,title_NO_SW_Stem,tag
            q_tmp = SO_Que(row[0], row[1], row[2], row[3], row[4], row[5])
            SO_datalist.append(q_tmp)
    except Exception as e:
        print e
    cur.close()
    con.close()
    return SO_datalist


def read_all_questions_from_post():
    sql = 'SELECT * FROM posts where Tags like \'%<java>%\' and AnswerCount > 0'
    SO_datalist = []
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    try:
        cur.execute(sql)
        results = cur.fetchall()
        count = 0
        for row in results:
            count += 1
            # id,type,title,title_NO_SW,title_NO_SW_Stem,tag
            q_tmp = SO_Que(row[0], row[1], row[2], row[3], row[4], row[5])
            SO_datalist.append(q_tmp)
    except Exception as e:
        print e
    cur.close()
    con.close()
    return SO_datalist


# repo
def read_all_questions_from_repo():
    sql = 'SELECT * FROM repo_qs'
    SO_datalist = []
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    try:
        cur.execute(sql)
        results = cur.fetchall()
        count = 0
        for row in results:
            count += 1
            # id,title,body,tag
            q_tmp = SO_Que(row[0], row[1], row[2], row[3])
            SO_datalist.append(q_tmp)
    except Exception as e:
        print e
    cur.close()
    con.close()
    return SO_datalist


def read_specific_question_from_repo(id):
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    sql = "SELECT * FROM repo_qs WHERE Id=" + str(id)
    try:
        cur.execute(sql)
        results = cur.fetchall()
        for row in results:
            # id,title,body,tag
            q_tmp = SO_Que(row[0], row[1], row[2], row[3])
    except Exception as e:
        print e
    cur.close()
    con.close()
    return q_tmp


def read_specific_question_from_post(id):
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    sql = "SELECT * FROM posts WHERE Id=" + str(id)
    try:
        cur.execute(sql)
        results = cur.fetchall()
        for row in results:
            # id,title,body,tag
            q_tmp = SO_Que()
    except Exception as e:
        print e
    cur.close()
    con.close()
    return q_tmp


def read_specific_question_from_post(id):
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    sql = "SELECT * FROM posts WHERE Id=" + str(id)
    try:
        cur.execute(sql)
        results = cur.fetchall()
        for row in results:
            # id,type,title,tag
            q_tmp = SO_Que(row[0], row[1], row[11], row[12])
            q_tmp = preprocessing_for_que(q_tmp)
    except Exception as e:
        print e
    cur.close()
    con.close()
    return q_tmp


def read_duplicate_pair_from_postlink_table(num):
    Duplicate_pair = []
    postlink_sql = "SELECT * FROM post_links ORDER BY PostId"
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    try:
        cur.execute(postlink_sql)
        results = cur.fetchall()
        count = 1
        for row in results:
            id1 = row[2]
            if ifjava_post(id1) == False:
                continue
            id2 = row[3]
            if ifjava_post(id2) == False:
                continue
            Duplicate_pair.append([id1, id2])
            print 'processing ' + str(count)
            if count >= num:
                break
            count += 1
    except Exception as e:
        print e
    cur.close()
    con.close()
    return Duplicate_pair


def ifjava_post(id):
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    sql = "SELECT * FROM posts WHERE Id=" + str(id) + " and Tags like '%<java>%'"
    ifjava = False
    try:
        cur.execute(sql)
        results = cur.fetchall()
        if len(results) != 0:
            ifjava = True
    except Exception as e:
        print e
    cur.close()
    con.close()
    return ifjava

# def read_correspond_answers_from_java_table(top_dq_id_and_sim):
#     corr_answers = {}
#     con = mdb.connect('localhost', 'root', '123456', 'answerbot')
#     cur = con.cursor()
#     try:
#         for (q_id, sim) in top_dq_id_and_sim:
#             corr_answer = []
#             sql = "SELECT * FROM java_ans WHERE PostTypeId = 2 AND ParentId = " + str(q_id)
#             cur.execute(sql)
#             results = cur.fetchall()
#             for row in results:
#                 # id, body, score, parent_id
#                 SO_AnswerUnit_tmp = SO_Ans(row[0], row[6], row[4], row[17])
#                 corr_answer.append(SO_AnswerUnit_tmp)
#             corr_answers[q_id] = corr_answer
#     except Exception as e:
#         print e
#     cur.close()
#     con.close()
#     return corr_answers
def read_correspond_answers_from_java_table(top_dq_id_and_sim,file_path):
    for (q_id, sim) in top_dq_id_and_sim:

        corr_answers = {}
        path = file_path
        with open(path, 'rb')as f:
            new_dict = pickle.load(f)
        corr_answer = []
        for order, item in enumerate(new_dict):
            SO_AnswerUnit_tmp = SO_Ans(item[0], item[3:], item[1], item[2])
            corr_answer.append(SO_AnswerUnit_tmp)
        corr_answers[q_id] = corr_answer
        # print(corr_answer)
    return corr_answers


def read_correspond_answer_from_java_table(q_id):
    corr_answer = []
    sql = "SELECT * FROM java_ans WHERE PostTypeId = 2 AND ParentId = " + str(q_id)
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    try:
        cur.execute(sql)
        results = cur.fetchall()
        for row in results:
            # id, body, score, parent_id
            SO_AnswerUnit_tmp = SO_Ans(row[0], row[6], row[4], row[17])
            corr_answer.append(SO_AnswerUnit_tmp)
    except Exception as e:
        print e
    cur.close()
    con.close()
    return corr_answer


def read_q_list_from_java(id_list):
    qlist = []
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    count = 1
    for qid in id_list:
        sql = "SELECT * FROM java_qs WHERE Id = %s" % qid[0]
        try:
            cur.execute(sql)
            results = cur.fetchall()
            row = results[0]
            # id,title,body,tags
            qlist.append(SO_Que(row[0], row[11], row[6], row[12]))
            count += 1
            if count % 1000 == 0:
                print 'reading ' + str(count) + ' question from Table java_qs'
        except Exception as e:
            print e
    cur.close()
    con.close()
    return qlist


def read_id_list(tablename):
    id_list = []
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    count = 1
    for id in id_list:
        if count % 10000 == 0:
            print 'reading ' + str(count) + ' question from Table java'
        count += 1
        sql = "SELECT Id FROM %s" % tablename
        try:
            cur.execute(sql)
            results = cur.fetchall()
            for row in results:
                # id,type,title,tag
                id_list.append(row[0])
        except Exception as e:
            print e
    cur.close()
    con.close()
    return id_list


# repo
def insert_qlist_to_table(qlist, tablename):
    print "start to insert...", get_current_time()
    con = mdb.connect('localhost', 'root', '123456', 'answerbot')
    cur = con.cursor()
    count = 1
    for q in qlist:
        try:
            # id,title,body,tag
            title = mdb.escape_string(q.title)
            body = mdb.escape_string(q.body)
            tag = mdb.escape_string(','.join(q.tag))
            sql = "INSERT INTO %s VALUES('%s', '%s', '%s', '%s')" % (tablename, q.id, title, body, tag)
            cur.execute(sql)
            con.commit()
            count += 1
            if count % 1000 == 0:
                print('Inserting ' + str(count) + ' question to Table %s' % tablename, get_current_time())
        except Exception as e:
            print "id %s\ntitle %s\nbody %s\ntag %s\n" % (q.id, q.title, q.body, q.tag)
            print e
    cur.close()
    con.close()
    print('Insert finished.', get_current_time())
    return
