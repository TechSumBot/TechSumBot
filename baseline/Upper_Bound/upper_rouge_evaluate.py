from lib2to3.pgen2.literals import test
from pyrouge import Rouge155
import os
import shutil
import re
import logging
import util.rouge_evaluator as rouge_wrapper
os.chdir('../../dataset')   #修改当前工作目录



def exception_test(num_annotator):
    '''
    if the rouge module drops the error, use this function to 
    iteratively detect if any gold summary contains the invalid charactors (e.g., <>)

    '''
    for i in range(37):
        for file in os.listdir('gold'):
            num_rule = '(?<=['+num_annotator+']_)[0-9]*'
            searchobj2 = re.search(num_rule,file)
            if searchobj2:
                flag =  searchobj2[0] 
                if flag!='':
                    if i == int(searchobj2[0]):
                        shutil.copy(os.path.join('gold',file), os.path.join('test',file))
                        file_2=str(2)+file[1:]
                        shutil.copy(os.path.join('gold',file_2), os.path.join('test',file_2))
                        file_3=str(3)+file[1:]
                        shutil.copy(os.path.join('gold',file_3), os.path.join('test',file_3))
                        try:
                            rouge_calculate()
                        except:
                            print(file)
                            print('this is the wrong file')
                            exit(0)


def inter_aggrement():
    '''
    calculate the rouge score for each annotator to others
    consider the avg scores as the inter-annotator agreement
    '''
    evaluator_1 = rouge_wrapper.RougeEvaluator(system_filename_pattern='[1]_(\d+)_[\s\S]*.txt',
                       model_filename_pattern='[123]_#ID#_[\s\S]*.txt',
                       system_dir = 'gold',
                       model_dir = 'gold' 
                       )
    output_1 = evaluator_1.evaluate()
    # print(output_1['short_output'])
    print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator_1.short_result(output_1))
    sys_1_1, sys_1_2, sys_1_3 = evaluator_1.short_result(output_1)

    evaluator_2 = rouge_wrapper.RougeEvaluator(system_filename_pattern='[2]_(\d+)_[\s\S]*.txt',
                       model_filename_pattern='[123]_#ID#_[\s\S]*.txt',
                       system_dir = 'gold',
                       model_dir = 'gold' 
                       )
    output_2 = evaluator_2.evaluate()
    # print(output_2['short_output'])
    print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator_2.short_result(output_2))
    sys_2_1, sys_2_2, sys_2_3 = evaluator_2.short_result(output_2)



    evaluator_3 = rouge_wrapper.RougeEvaluator(system_filename_pattern='[3]_(\d+)_[\s\S]*.txt',
                       model_filename_pattern='[123]_#ID#_[\s\S]*.txt',
                       system_dir = 'gold',
                       model_dir = 'gold' 
                       )
    output_3 = evaluator_3.evaluate()
    # print(output_3['short_output'])
    # print(evaluator_3.short_result(output_3))
    print('The rouge-1 of evaluator 3 is %s and the rouge-2 of evaluator 3 is %s and the rouge-L of evaluator 3 is %s'%evaluator_3.short_result(output_3))
    sys_3_1, sys_3_2, sys_3_3 = evaluator_3.short_result(output_3)

    print((float(sys_1_1)+float(sys_2_1)+float(sys_3_1))/3)
    print((float(sys_1_2)+float(sys_2_2)+float(sys_3_2))/3)
    print((float(sys_1_3)+float(sys_2_3)+float(sys_3_3))/3)


    # re_rule = '(?<=ROUGE-[12L]\s)[0-9\.]{7}(?=\s)'



def main():
    inter_aggrement()

if __name__ =='__main__':
    main()

