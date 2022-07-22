# -*- coding: UTF-8 -*-

def remove_duplicate_element(list_tmp):
    clean_list = []
    for word in list_tmp:
        if word not in clean_list:
            clean_list.append(word)
    return clean_list


def get_dic_tf_from_list(word_list):
    dic = {}
    len_of_list = len(word_list)
    for word in word_list:
        if word not in dic:
            dic[word] = 1.0
        else:
            dic[word] = dic[word] + 1.0
    for key in dic.keys():
        dic[key] = dic[key] / len_of_list
    return dic


def merge_list(list1, list2):
    all_list = []
    for ele in list1:
        if ele not in all_list:
            all_list.append(ele)
    for ele in list2:
        if ele not in all_list:
            all_list.append(ele)
    return all_list

