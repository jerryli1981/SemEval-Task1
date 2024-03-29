#pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment

import os
import glob
import re

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data')
pastData_dir = os.path.join(base_dir, 'pastData')

gs = glob.glob(os.path.join(pastData_dir, '*/STS.gs.*.txt'))
inputs = glob.glob(os.path.join(pastData_dir, '*/STS.input.*.txt'))

id = 1

with open(os.path.join(data_dir, 'SICK_train_rich.txt'), 'w') as f:
	f.write("pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment\n")
	for g, i in zip(gs,inputs):
 		with open(g, 'r') as gf, open(i, 'r') as iptf:
 			for score, line in zip(gf, iptf):
 				if not re.match(r'\w+', score):
 					continue
 				line = re.sub("#", "", line)
 				a,b = line.strip().split('\t')
 				f.write(str(id)+"\t"+a+"\t"+b+"\t"+score.strip()+"\t"+"ENTAILMENT\n")
 				id += 1


with open(os.path.join(data_dir, 'SICK_trial_big.txt'), 'w') as f1, \
	open(os.path.join(data_dir, 'SICK_train_big.txt'), 'w') as f2:
 	f1.write("pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment\n")
 	f2.write("pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment\n")
 
 	with open(os.path.join(data_dir, 'SICK_train_rich.txt'), 'rb') as f:
 		f.readline()
 		for i, line in enumerate(f):
 			if i % 20 == 0:
 				f1.write(line)
 			else:
 				f2.write(line)
								
test_dir = os.path.join(base_dir, 'sts2016-english-v1.1')
test_fs = ["answer-answer", "headlines", "plagiarism", "postediting", "question-question"]

for fn in test_fs:

	id = 1

	with open(os.path.join(data_dir, 'SICK_test_'+fn+'.txt'), 'w') as f1, \
		open(os.path.join(test_dir, 'STS2016.input.'+fn+'.txt'), 'r') as f2:
		f1.write("pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment\n")

		for line in f2:
			first, second, _,_ = line.split('\t')

			f1.write(str(id) + "\t" + first + "\t" + second + "\t" + "NA" + "\t" + "NA" +"\n")
			id += 1









