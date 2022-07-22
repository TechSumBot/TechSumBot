from lib2to3.pgen2.literals import test
from pyrouge import Rouge155
import os
import shutil
import re
import logging
from lib2to3.pgen2.literals import test
from pyrouge import Rouge155
import os
import shutil
import re
import logging


class RougeEvaluator():
    '''Wrapper for pyrouge'''
    def __init__(self, system_filename_pattern='[123]_(\d+)_[\s\S]*.txt',
                       model_filename_pattern='[123]_#ID#_[\s\S]*.txt',
                       system_dir=None, model_dir=None,
                       log_level=logging.WARNING):
        self.system_dir = system_dir
        self.model_dir = model_dir

        self.r = Rouge155()
        self.r.log.setLevel(log_level)
        self.r.system_filename_pattern = system_filename_pattern
        self.r.model_filename_pattern = model_filename_pattern

        self.results_regex = \
                re.compile('(ROUGE-[12L]) Average_F: ([0-9.]*) \(95.*?([0-9.]*) - ([0-9.]*)')

    def evaluate(self, system_dir=None, model_dir=None):
        if system_dir is None:
            assert self.system_dir is not None, 'no system_dir given'
            system_dir = self.system_dir
        if model_dir is None:
            assert self.model_dir is not None, 'no model_dir given'
            model_dir = self.model_dir

        self.r.system_dir = system_dir
        self.r.model_dir = model_dir

        full_output = self.r.convert_and_evaluate()
        results = self.results_regex.findall(full_output)

        outputs = {}
        outputs['full_output'] = full_output
        outputs['dict_output'] = self.r.output_to_dict(full_output)
        outputs['short_output'] = '\n'.join(['  {0} {1} ({2} - {3})'.format(*r) for r in results])

        return outputs

    def short_result(self,output):
        rouge1_rule = '(?<=ROUGE-[1]\s)[0-9\.]{7}(?=\s)'
        searchobj1 = re.search(rouge1_rule,output['short_output'])
        rouge_1 = searchobj1[0]
        rouge2_rule = '(?<=ROUGE-[2]\s)[0-9\.]{7}(?=\s)'
        searchobj2 = re.search(rouge2_rule,output['short_output'])
        rouge_2 = searchobj2[0]        
        rougeL_rule = '(?<=ROUGE-[L]\s)[0-9\.]{7}(?=\s)'
        searchobjL = re.search(rougeL_rule,output['short_output'])
        rouge_L = searchobjL[0]          
        return rouge_1, rouge_2, rouge_L

def inter_aggrement():
    '''
    calculate the rouge score for the answerbot output to the groundtruth
    '''
    evaluator = RougeEvaluator(system_filename_pattern='(\d+)_[\s\S]*.txt',
                       model_filename_pattern='[123]_#ID#_[\s\S]*.txt',
                       system_dir = 'result/textrank_simcse_1_top20_asnq',
                       model_dir = '../dataset/gold' 
                       )
    output_1 = evaluator.evaluate()
    # print(output_1['short_output'])
    print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator.short_result(output_1))

def inter_aggrement_grid_search(system_dir):
    '''
    calculate the rouge score for the answerbot output to the groundtruth
    '''
    evaluator = RougeEvaluator(system_filename_pattern='(\d+)_[\s\S]*.txt',
                       model_filename_pattern='[123]_#ID#_[\s\S]*.txt',
                       system_dir = system_dir,
                       model_dir = '../dataset/gold' 
                       )
    output_1 = evaluator.evaluate()
    # print(output_1['short_output'])
    # print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator.short_result(output_1))
    if float(evaluator.short_result(output_1)[2])>0.52:
        print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator.short_result(output_1))
        print(system_dir)
        print('===================\n\n\n')

def grid_search_inter_aggrement():
    data_dir = 'result'
    for item in os.listdir(data_dir):
        inter_aggrement_grid_search('result/'+item)


def main():
    inter_aggrement()
    # grid_search_inter_aggrement()

if __name__ =='__main__':
    main()

