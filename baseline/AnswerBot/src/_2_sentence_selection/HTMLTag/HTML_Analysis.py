# -*- coding: UTF-8 -*-


def get_html_score(text):
    HTMLTag_plus = ['<strong>', '<code>']
    HTMLTag_minus = ['<strike>']
    len_of_tag_plus = len(HTMLTag_plus)
    score = 1.0
    for pattern in HTMLTag_plus:
        if pattern in text.lower():
            score += (1.0 / len_of_tag_plus)
    for pattern in HTMLTag_minus:
        if pattern in text.lower():
            score = 1.0
            break
    return score
