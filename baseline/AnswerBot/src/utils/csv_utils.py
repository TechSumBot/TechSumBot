# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     csv_utils
   Description :
   Author :       xubowen
   date：          2018/10/31 11:39 AM
-------------------------------------------------
   Change Activity:
                   2018/10/31 11:39 AM
-------------------------------------------------
"""

import csv


def write_Q_list_to_csv(q_list, csv_fpath, header):
    print("# q_list : %s" % len(q_list))
    with open(csv_fpath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(header)
        for q in q_list:
            wr.writerow([q.qid, q.title, q.desc_text, q.desc_code, q.tags])
    print("Write %s successfully!" % csv_fpath)


def read_csv_to_dict(csv_fpath):
    import pandas as pd
    return pd.Series.from_csv(csv_fpath, header=0).to_dict()


def write_dict_to_csv(dict_tmp, csv_fpath, header):
    with open(csv_fpath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(header)
        for k in dict_tmp:
            try:
                wr.writerow([k, dict_tmp[k]])
            except Exception as e:
                print("Error %s" % e)
    print("Write %s successfully!" % csv_fpath)


def write_list_to_csv(list_tmp, csv_fpath, header):
    with open(csv_fpath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(header)
        for x in list_tmp:
            try:
                wr.writerow(x)
            except Exception as e:
                print("Error %s" % e)
    print("Write %s successfully!" % csv_fpath)

def write_list_to_txt(list_tmp, csv_fpath):
    with open(csv_fpath, 'w') as myfile:
        myfile.write(list_tmp[0][1])
    print("Write %s successfully!" % csv_fpath)

def load_csv2dict(csv_fpath):
    mydict = {}
    with open(csv_fpath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for rows in reader:
            k = rows[0]
            v = rows[1]
            mydict[k] = v
    return mydict


def load_vocab_from_csv(vocab_fpath):
    mydict = {}
    print("loading vocab %s" % vocab_fpath)
    with open(vocab_fpath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for rows in reader:
            k = rows[0]
            v = rows[1]
            mydict[k] = v
    return mydict


def load_tag_vocab_from_csv(tag_vocab_fpath):
    tags = []
    print("loading tag vocab %s" % tag_vocab_fpath)
    with open(tag_vocab_fpath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for rows in reader:
            tags.append(rows[0])
    return tags
