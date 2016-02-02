#pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment

import os
import glob
import re

#f.write(str(id)+"\t"+a+"\t"+b+"\t"+score.strip()+"\t"+"ENTAILMENT\n")

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data')
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


