# -*- coding: UTF-8 -*-


def get_order_score(order):
    return Segmentation_function(order)


def Segmentation_function(order):
    if order == 1:
        return 2.0
    elif order == 2:
        return 1.5
    elif order == 3:
        return 1.33
    else:
        return 1.0
