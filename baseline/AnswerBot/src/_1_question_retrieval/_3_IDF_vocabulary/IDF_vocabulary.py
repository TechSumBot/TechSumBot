# -*- coding: utf-8 -*-
import os
import sys
root_path = "/home/hywang/answerbot-tool/src"
sys.path.append(root_path)
from utils.db_util import read_all_questions_from_repo
import math
from utils.time_utils import get_current_time
from nltk import word_tokenize
from utils.csv_utils import write_list_to_csv
import operator


def build_IDF_vocabulary():
    qlist = read_all_questions_from_repo()
    total_num = len(qlist)
    voc = {}
    count = 0
    for q in qlist:
        title_wlist = word_tokenize(q.title.strip())
        cur_word_set = set()
        for w in title_wlist:
            if w not in cur_word_set:
                cur_word_set.add(w)
                if w not in voc.keys():
                    voc[w] = 1.0
                else:
                    voc[w] = voc[w] + 1.0

        # body_wlist = word_tokenize(q.body.strip())
        # for w in body_wlist:
        #     if w not in cur_word_set:
        #         cur_word_set.add(w)
        #         if w not in voc.keys():
        #             voc[w] = 1.0
        #         else:
        #             voc[w] = voc[w] + 1.0

        count += 1
        if count % 10000 == 0:
            print 'processing %s unit...' % count, get_current_time()
    for key in voc.keys():
        idf = math.log(total_num / (voc[key] + 1.0))
        voc[key] = idf
    sorted_voc = sorted(voc.items(), key=operator.itemgetter(1))
    return sorted_voc





if __name__ == '__main__':
    fpath = 'idf_vocab.csv'
    header = ['word', 'idf']
    vocab = build_IDF_vocabulary()
    write_list_to_csv(vocab, fpath, header)
    print 'Done.'
