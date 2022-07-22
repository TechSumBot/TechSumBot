# -*- coding: utf-8 -*-
# import MySQLdb as mdb
from utils.file_util import write_file
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path_of_dic = os.path.join(dir_path, 'entity_dic.txt')


def load_entity_set():
    dic = set()
    for line in open(path_of_dic):
        line = line.strip()
        dic.add(line)
    return dic


def extract_tag_info_from_java_table():
    sql = 'SELECT * FROM java_qs'
    con = mdb.connect('localhost', 'root', 'root', 'answerbot')
    cur = con.cursor()
    dic = set()
    try:
        cur.execute(sql)
        results = cur.fetchall()
        cnt = 0
        for row in results:

            # tag : '<java><xml><csv><data-conversion>'
            tag_list_tmp = row[12].replace('<', ' ').replace('>', ' ').replace('  ', ' ').strip()
            for tag_tmp in tag_list_tmp.split(' '):
                if tag_tmp not in dic:
                    dic.add(tag_tmp)
            cnt += 1
            if cnt % 1000 == 0:
                print 'processing ' + str(cnt) + ' instance'
    except Exception as e:
        print e
    con.close()
    return dic


if __name__ == '__main__':
    path_of_dic = 'entity_dic.txt'
    dic = extract_tag_info_from_java_table()
    dic_str = ''
    for tag_tmp in dic:
        dic_str += (tag_tmp + '\n')
    write_file(path_of_dic, dic_str)
    print 'Done.'
