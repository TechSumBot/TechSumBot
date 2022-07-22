# -*- coding: UTF-8 -*-

from utils.time_utils import get_current_time
import pymysql as mdb
from utils.csv_utils import write_list_to_csv
import pandas as pd

"""
Before execute this script, please execute following sql script.
Get java-qid_list.csv to make sure related posts exist in data repository

SELECT Id FROM java_qs WHERE PostTypeId = 1 INTO OUTFILE '/var/lib/mysql-files/java_qid_list.csv' FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';
"""


def load_java_qid_set(csv_fpath):
    id_set = set()
    df = pd.read_csv(csv_fpath, header=None)
    for idx, row in df.iterrows():
        id_set.add(row[0])
    print('#java questions = %s' % len(id_set), get_current_time())
    return id_set


def extract_java_relevant_ids_from_postlink(java_id_set):
    sql = "SELECT * FROM post_links"
    con = mdb.connect('localhost', 'root', 'root', '05-Sep-2018-SO')
    cur = con.cursor()
    id_dict = {}
    cnt = 0
    try:
        cur.execute(sql)
        results = cur.fetchall()
        for row in results:
            postId = row[2]
            related_postId = row[3]
            if postId in java_id_set and related_postId in java_id_set:
                if postId not in id_dict:
                    id_dict[postId] = True
                if related_postId not in id_dict:
                    id_dict[related_postId] = True
            cnt += 1
            if cnt % 10000 == 0:
                print('Processing %s...' % cnt, get_current_time())
    except Exception as e:
        print e
    cur.close()
    con.close()
    print("# relevant qid = %s" % len(id_dict), get_current_time())
    return sorted(list(id_dict.keys()))


if __name__ == '__main__':
    java_qid_set_fpath = 'java_qid_list.csv'
    java_id_set = load_java_qid_set(java_qid_set_fpath)
    related_id_list = extract_java_relevant_ids_from_postlink(java_id_set)
    # post id list
    related_id_list_fpath = 'related_qid_list.txt'
    header = ['Id']
    write_list_to_csv(related_id_list, related_id_list_fpath, header)
