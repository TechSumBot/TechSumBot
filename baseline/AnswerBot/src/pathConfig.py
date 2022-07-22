# -*- coding: UTF-8 -*-

import os

project_dir = 'Your path'
res_dir = os.path.join(project_dir, "res")


def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))
