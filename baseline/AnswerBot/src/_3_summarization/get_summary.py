import os
import sys
root_path = "/home/hywang/answerbot-tool/src"
sys.path.append(root_path)
from _3_summarization.MMR_Analysis import MMR_Analysis
from pathConfig import res_dir
import os
from utils.data_util import preprocessing_for_query
from utils.csv_utils import write_list_to_csv


def get_summary(query, top_ss, topk):
    selected_sentence = MMR_Analysis(query, top_ss, topk)
    summary = '\n'.join([x for x in selected_sentence])
    return summary

# def get_summary_modified(query, top_ss, topk):
#     selected_sentence = MMR_Analysis(query, top_ss, topk)
#     # summary = '\n'.join([x for x in selected_sentence])
#     for sent in selected_sentence:
#         summary.append()
#     return summary

def load_ss_result(ss_fpath):
    import pandas as pd
    ss_res = list()
    df = pd.read_csv(ss_fpath)
    for idx, row in df.iterrows():
        ss_res.append((row[0], eval(row[1])))
    return ss_res


if __name__ == '__main__':
    ss_fpath = os.path.join(res_dir, 'ss_res.csv')

    topk = 5
    res = list()
    # print load_ss_result(ss_fpath)
    for query, ss in load_ss_result(ss_fpath):
        query = ' '.join(preprocessing_for_query(query))
        sum = get_summary(query, ss, topk)
        res.append([query, sum])
        print("summary\n%s" % sum)

    res_fpath = os.path.join(res_dir, 'summary_res.csv')
    header = ["query", "summary"]
    write_list_to_csv(res, res_fpath, header)
