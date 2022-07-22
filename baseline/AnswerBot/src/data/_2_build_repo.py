# -*- coding: UTF-8 -*-

from utils.db_util import read_q_list_from_java, insert_qlist_to_table
import pandas as pd
from utils.preprocessing_util import preprocessing_for_que
from utils.time_utils import get_current_time
import sys

reload(sys)
sys.setdefaultencoding('utf8')

"""
CREAT repo

CREATE TABLE repo_qs (
    Id INT NOT NULL PRIMARY KEY,
    Title text NULL,
    Body text NULL,
    Tags VARCHAR(256)
);

create index repo_qs_id on repo_qs(Id);
"""


def preprocessing(qlist):
    print "preprocessing...", get_current_time()
    for i in range(len(qlist)):
        qlist[i] = preprocessing_for_que(qlist[i])
        if i % 1000 == 0:
            print "preprocessing %s question..." % i, get_current_time()
    return qlist


if __name__ == '__main__':
    related_id_fpath = "related_qid_list.txt"
    related_id_list = pd.read_csv(related_id_fpath).values
    qlist = read_q_list_from_java(related_id_list)
    qlist = preprocessing(qlist)
    tablename = 'repo_qs'
    insert_qlist_to_table(qlist, tablename)
