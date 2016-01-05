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
				a,b = line.strip().split('\t')
				f.write(str(id)+"\t"+a+"\t"+b+"\t"+score.strip()+"\t"+"ENTAILMENT\n")
				id += 1









