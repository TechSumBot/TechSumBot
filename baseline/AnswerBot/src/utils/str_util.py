import unicodedata

# -*- coding: UTF-8 -*-
ch_set = ['.', '?', '!']


def split_into_sentence(text):
    text = text.replace('<p>', ' ').replace('</p>', ' ').replace('  ', ' ')
    sentences = []
    size = len(text)
    sent = ''
    i = 0
    while i < size:
        if text[i] in ch_set:
            if i != size - 1:
                if text[i + 1] == ' ':
                    # e.g.
                    if i > 3 and text[i - 1].lower() == 'g' and text[i - 2].lower() == '.' and text[
                        i - 3].lower() == 'e':
                        pass
                    # vs.
                    if i > 2 and text[i - 2].lower() == 'v' and text[i - 1].lower() == 's':
                        pass
                    else:
                        sentences.append(sent)
                        sent = ''
                        i += 1
                else:
                    sent += text[i]
            else:
                sentences.append(sent)
                sent = ''
        else:
            sent += text[i]
        i += 1
    return sentences


def split_into_paragraph(text):
    # extract <p>sent</p>
    paragraph_list = []
    tag_head = '<p>'
    tag_tail = '</p>'
    while tag_head in text:
        head_pos = text.find(tag_head)
        tail_pos = text.find(tag_tail)
        if head_pos >= tail_pos:
            break
        ahref_head = text[head_pos:tail_pos].find(">")
        tag_content = text[head_pos + ahref_head + 1:tail_pos]
        text = text[:head_pos] + text[tail_pos + len(tag_tail):]
        if tag_content != '':
            paragraph_list.append(tag_content)
    return paragraph_list


def unicode2str(text):
    if isinstance(text, unicode):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')


if __name__ == '__main__':
    text = 'java.jdk.api. I like apple? yes, I like it! Oh, I like it.'
    for sent in split_into_sentence(text):
        print sent
