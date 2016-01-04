1. mkdir data, put SICK_train.txt, SICK_dev.txt, SICK_test.txt into data dir
2. run python preprocess-sick.py to do NLP
3. mkdir data/glove, than run convert-wordvecs.sh
4. run th main.lua to train/test/predict model