# -*- coding: UTF-8 -*-


def get_pattern_score(text):
    Pattern = ['please check', 'pls check', 'you should', 'you can try', 'you could try', 'check out',
               'in short', 'the most important is', 'I d recommend', 'in summary', 'keep in mind that',
               'i suggest that']
    for pattern in Pattern:
        if pattern in text.lower():
            return 2.0
    return 1.0


if __name__ == '__main__':
    text = 'Jupiter supports up to Eclipse 3.5. Check out Jupiter Downloads Page '
    print get_pattern_score(text)
