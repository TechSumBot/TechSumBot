from lib2to3.pgen2.literals import test
from pyrouge import Rouge155
import os
import shutil
import re
import logging
import util.rouge_evaluator as rouge_wrapper


def inter_aggrement():
    '''
    calculate the rouge score for the answerbot output to the groundtruth
    '''
    evaluator = rouge_wrapper.RougeEvaluator(system_filename_pattern='(\d+)_[\s\S]*.txt',
                       model_filename_pattern='[123]_#ID#_[\s\S]*.txt',
                       system_dir = 'result/',
                       model_dir = '../../../../dataset/gold' 
                       )
    output_1 = evaluator.evaluate()
    # print(output_1['short_output'])
    print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator.short_result(output_1))


inter_aggrement()