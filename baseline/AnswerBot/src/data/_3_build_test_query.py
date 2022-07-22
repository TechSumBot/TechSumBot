# -*- coding: UTF-8 -*-

from utils.file_util import read_sentence_by_line
from utils.Random_util import get_random_list
from dataset_util import get_high_relevant_questions, read_post_pair
from utils.db_util import read_specific_question_from_repo
from utils.file_util import write_file


'''
Attention : PLZ DON'T RERUN THIS CODE! OR ALL THE DATA ARE INVALID!
'''

if __name__ == '__main__':
    path_of_post_id_list = 'post_id_list.txt'
    id_list = read_sentence_by_line(path_of_post_id_list)
    path_of_post_pair = 'post_pair_list.txt'
    pair_list = read_post_pair(path_of_post_pair)
    testnum = 100
    size = len(id_list)
    random_list = get_random_list(0, size, testnum)
    dic = {}
    for random_num in random_list:
        id = id_list[random_num]
        high_relevant_id_list = get_high_relevant_questions(id, pair_list)
        dic[id] = high_relevant_id_list
    write_str = ''
    for id in dic.keys():
        # query id
        # query
        # high relevant id list
        write_str += str(id) + '\n'
        write_str += (str(read_specific_question_from_repo(id).title) + '\n')
        for relevant_id in dic[id]:
            write_str += (str(relevant_id) + ' ')
        write_str = write_str.strip() + '\n'
    path_of_query = 'query.txt'
    write_file(path_of_query, write_str)
    print 'Done.'
