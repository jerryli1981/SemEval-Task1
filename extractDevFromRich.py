#pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment

import os
import glob
import re

#f.write(str(id)+"\t"+a+"\t"+b+"\t"+score.strip()+"\t"+"ENTAILMENT\n")

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data')

count = 0

with open(os.path.join(data_dir, 'SICK_trial_big.txt'), 'w') as f1, \
	open(os.path.join(data_dir, 'SICK_train_big.txt'), 'w') as f2, \
	f1.write("pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment\n")
	f2.write("pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment\n")

	with open(os.path.join(data_dir, 'SICK_train_rich.txt'), 'rb') as f:
		f.readline()
		for i, line in enumerate(f):
			if i % 20 == 0:
				f1.write(line)
			else:
				f2.write(line)