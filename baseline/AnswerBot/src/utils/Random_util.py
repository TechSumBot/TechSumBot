# -*- coding: UTF-8 -*-

import random


def get_random_list(st, et, num):
    list = range(st, et)
    random_list = random.sample(list, num)
    return random_list
