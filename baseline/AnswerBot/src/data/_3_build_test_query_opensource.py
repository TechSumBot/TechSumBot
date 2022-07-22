# -*- coding: UTF-8 -*-

from utils.file_util import read_sentence_by_line
from utils.Random_util import get_random_list
from dataset_util import get_high_relevant_questions, read_post_pair
from utils.db_util import read_specific_question_from_repo
from utils.file_util import write_file

if __name__ == '__main__':
    path_of_file = 'query.txt'
    file = open(path_of_file)
    linenum = 1
    write_str = ''
    for line in file:
        # query id
        # query
        # high relevant id list
        line = line.strip()
        if linenum % 3 != 0:
            write_str += line + '\n'
        linenum += 1
    path_of_query = 'open_query.txt'
    write_file(path_of_query, write_str.strip())
    print 'Done.'
