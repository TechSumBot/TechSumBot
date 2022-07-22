# -*- coding: UTF-8 -*-

import copy
from pathConfig import get_base_path


def read_post_pair(path):
    file = open(path)
    pair = []
    for line in file:
        line = line.strip().split(' ')
        pair.append([line[0], line[1], line[2]])
    return pair


def get_high_relevant_questions(id, pair_list):
    set = []
    for [id1, id2, type] in pair_list:
        # direct link
        if id1 == id and type == '1':
            if id2 not in set:
                set.append(id2)
        elif id2 == id and type == '1':
            if id1 not in set:
                set.append(id1)
        # duplicate link
        elif id1 == id and type == '3':
            if id2 not in set:
                set.append(id2)
                new_pair_list = copy.deepcopy(pair_list)
                new_pair_list.remove([id1, id2, type])
                set += get_high_relevant_questions(id2, new_pair_list)
        elif id2 == id and type == '3':
            if id1 not in set:
                set.append(id1)
                new_pair_list = copy.deepcopy(pair_list)
                new_pair_list.remove([id1, id2, type])
                set += get_high_relevant_questions(id1, new_pair_list)
    return set


def read_query_for_testing():
    path_of_query = get_base_path() + '/data/query.txt'
    file = open(path_of_query)
    count = 0
    query_list = []
    query_id = -1
    query = ''
    for line in file:
        line = line.strip()
        if count % 3 == 0:
            query_id = line
        elif count % 3 == 1:
            query = line
        else:
            relevant_id_list = line.split(' ')
            query_list.append([query_id, query, relevant_id_list])
        count += 1
    return query_list


def read_all_id_list():
    path_of_id_list = get_base_path() + '/data/post_id_list.txt'
    file = open(path_of_id_list)
    id_list = []
    for line in file:
        line = line.strip()
        id_list.append(line)
    return id_list


if __name__ == '__main__':
    pair = [[1, 3, '1'], [1, 4, '1'], [4, 5, '3'], [6, 5, '3'], [6, 7, 1], [7, 8, 1]]
    id = 5
    print get_high_relevant_questions(id, pair)
