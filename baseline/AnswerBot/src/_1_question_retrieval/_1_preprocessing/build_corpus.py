# -*- coding: utf-8 -*-
import os
import sys
root_path = 'your path'
sys.path.append(root_path)
from data_structure.SO_que import SO_Que
import pymysql as mdb
from utils.preprocessing_util import preprocessing_for_que
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def read_questions_from_java():
    sql = 'SELECT * FROM SO.java_qs;'
    con = mdb.connect('localhost', 'root', '123456', 'SO')
    cur = con.cursor()
    qlist = []
    cur.execute(sql)
    results = cur.fetchall()
    count = 0
    for row in results:
        count += 1
        # id,title,body,tag
        q_tmp = SO_Que(row[0], row[11], row[6], row[12])
        q_tmp = preprocessing_for_que(q_tmp)
        qlist.append(q_tmp)
        if len(qlist) % 10000 == 0:
            print 'Load %s questions...' % len(qlist)
    cur.close()
    con.close()
    return qlist


if __name__ == '__main__':
    corpus_fpath = 'corpus.txt'
    qlist = read_questions_from_java()
    with open(corpus_fpath, "w") as f:
        for q in qlist:
            f.write(q.title + " " + q.body + "\n")
    print 'Done.'
