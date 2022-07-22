# -*- coding: UTF-8 -*-
import os
import shutil


def write_file(filepath, strTmp):
    output = open(filepath, 'w')
    output.write(strTmp.strip())
    output.close()


def read_IdList(path):
    file = open(path)
    idList = []
    for line in file:
        line = line.strip().split(' ')
        for id in line:
            if id not in idList:
                idList.append(id)
    return idList


def read_group(path):
    file = open(path)
    group = []
    for line in file:
        line = line.strip().split(' ')
        group.append(line)
    return group


def read_index(path):
    file = open(path)
    index = []
    for line in file:
        line = line.strip().split(' ')
        index.append(line)
    return index


def read_sentence_by_line(path):
    file = open(path)
    sentences = []
    for line in file:
        line = line.strip()
        sentences.append(line)
    return sentences


def read_training_Pair(path):
    file = open(path)
    pair = []
    for line in file:
        line = line.strip().split(' ')
        pair.append(line)
    return pair


def read_training_pair_indexed(path):
    file = open(path)
    pair = []
    for line in file:
        line = line.strip().split(' ')
        pair.append(line)
    return pair


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("/")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # print path + ' 创建成功'
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print path + ' 目录已存在'
        return False


def moveFileto(sourceDir, targetDir):
    shutil.copy(sourceDir, targetDir)
