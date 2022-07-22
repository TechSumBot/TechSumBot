import os 
import random
import csv


querysum_diversity_gold = 0
querysum_usefulness_gold = 0

answerbot_diversity_gold = 0
answerbot_usefulness_gold = 0

sotasum_diversity_gold = 0
sotasum_usefulness_gold = 0

ground_truth = []
with open (os.path.join('groundtruth.csv'),'r') as f:
    reader = csv.reader(f)
    for order, item in enumerate(reader):
        for approach in item:
            if 'answer' in approach:
                ground_truth.append(0)
            if 'querysum' in approach:
                ground_truth.append(1)
            if 'approach' in approach:
                ground_truth.append(2)
print(len(ground_truth))
# for file in random.sample(os.listdir('csv'), 5):
for file in os.listdir('csv'):
    each_diversity = 0
    each_usefulness = 0
    with open (os.path.join('csv',file),'r') as f:
        reader = csv.reader(f)
        for order, item in enumerate(reader):
            # print(order)
            # for sent in item:

            if ground_truth[order] == 0:
                # print(item)
                answerbot_diversity_gold += int(item[0][-1])
                answerbot_usefulness_gold += int(item[1][-1])
            if ground_truth[order] == 1:
                querysum_diversity_gold += int(item[0])
                querysum_usefulness_gold += int(item[1])
            if ground_truth[order] == 2:
                sotasum_diversity_gold += int(item[0])
                each_diversity+=int(item[0])
                sotasum_usefulness_gold += int(item[1])
                each_usefulness+=int(item[1])

        print(file)
        # print('the average diversity and usefulness of sotasum are %f and %f'%(each_diversity/10,each_usefulness/10))


print('the average diversity and usefulness of answerbot are %f and %f'%(answerbot_diversity_gold/50,answerbot_usefulness_gold/50))
print('the average diversity and usefulness of querysum are %f and %f'%(querysum_diversity_gold/50,querysum_usefulness_gold/50))
print('the average diversity and usefulness of sotasum are %f and %f'%(sotasum_diversity_gold/50,sotasum_usefulness_gold/50))