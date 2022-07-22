# -*- coding: UTF-8 -*-

from nltk import word_tokenize
from utils.str_util import unicode2str


# question unit
def preprocessing_for_que(q):
    q.title = preprocess_title(q.title)
    q.body = preprocess_body(q.body)
    q.tag = preprocessing_for_tag(q.tag)
    return q


def preprocessing_for_tag(tag_str):
    return tag_str.replace('<', ' ').replace('>', ' ').strip().split()


def preprocess_title(title):
    text = title.lower()
    text = replace_double_space(text.replace('\n', ' '))
    text = tokenize_and_rebuild(text)
    return text.strip()


def preprocess_body(body):
    text = body.lower()
    text = remove_text_code(text)
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    text = tokenize_and_rebuild(text)
    if text:
        return text.strip()
    return ''


def remove_text_code(html_str):
    import re
    # regex: <pre(.*)><code>([\s\S]*?)</code></pre>
    regex_pattern = r'<pre(.*?)><code>([\s\S]*?)</code></pre>'
    html_text = html_str
    for m in re.finditer(regex_pattern, html_str):
        raw_code = html_str[m.start():m.end()]
        # remove code
        html_text = html_text.replace(raw_code, " ")
    return html_text.replace('\n', ' ')


# answer unit
def preprocessing_for_ans(ans):
    text = remove_text_code(ans.body.lower())
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    ans.body = text.strip()
    return ans


def preprocessing_for_ans_sent(sent):
    text = remove_text_code(sent.lower())
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    return text.strip()


def replace_double_space(text):
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text


def remove_html_tags(raw_html):
    from bs4 import BeautifulSoup
    try:
        text = BeautifulSoup(raw_html, "html.parser").text
    except Exception as e:
        # UnboundLocalError
        text = clean_html_tags2(raw_html)
    finally:
        return text.encode('utf8')


def clean_html_tags2(raw_html):
    import re
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def tokenize_and_rebuild(sent):
    sent = unicode(sent, 'utf-8')
    ws = word_tokenize(sent)
    return unicode2str(' '.join(ws))


if __name__ == '__main__':
    text = 'text > <img src= http://i.stack.imgur.com/ltCod.    png alt= Rich task editor >   '
    print remove_html_tags(text)
    print replace_double_space(text)
