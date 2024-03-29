"""
Preprocessing script for SICK data.

"""

import os
import glob

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)


def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
         open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile, \
         open(os.path.join(dst_dir, 'sim.txt'), 'w') as simfile,\
         open(os.path.join(dst_dir, 'ent.txt'), 'w') as entfile:
            datafile.readline()
            for line in datafile:
                i, a, b, sim, ent = line.strip().split('\t')
                afile.write(a+'\n')
                bfile.write(b+'\n')
                simfile.write(sim+'\n')
                entfile.write(ent+'\n')


def parse(dirpath, cp=''):

    dependency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=True)
    dependency_parse(os.path.join(dirpath, 'b.txt'), cp=cp, tokenize=True)

def build_word2Vector(glove_path, data_dir, vocab_name):

    print "building word2vec"
    from collections import defaultdict
    import numpy as np
    words = defaultdict(int)

    vocab_path = os.path.join(data_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["<UNK>"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        vocab[word] = idx

    print "word size", len(words)
    print "vocab size", len(vocab)


    word_embedding_matrix = np.zeros(shape=(300, len(vocab)))  

    
    import gzip
    wordSet = defaultdict(int)

    with open(glove_path, "rb") as f:
        for line in f:
           toks = line.split(' ')
           word = toks[0]
           if word in vocab:
                wordIdx = vocab[word]
                word_embedding_matrix[:,wordIdx] = np.fromiter(toks[1:], dtype='float32')
                wordSet[word] +=1
    
    count = 0   
    for word in vocab:
        if word not in wordSet:
            print word
            wordIdx = vocab[word]
            count += 1
            word_embedding_matrix[:,wordIdx] = np.random.uniform(-0.05,0.05, 300) 
    
    print "Number of words not in glove ", count
    import cPickle as pickle
    with open(os.path.join(data_dir, 'word2vec.bin'),'w') as fid:
        pickle.dump(word_embedding_matrix,fid)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing SICK dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    lib_dir = os.path.join(base_dir, 'lib')

    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    test_dir = os.path.join(data_dir, 'test')

    test_dir_answer_answer = os.path.join(test_dir, "answer-answer")
    test_dir_headlines = os.path.join(test_dir, "headlines")
    test_dir_plagiarism = os.path.join(test_dir, "plagiarism")
    test_dir_postediting = os.path.join(test_dir, "postediting")
    test_dir_question_question = os.path.join(test_dir, "question-question")

    make_dirs([train_dir, dev_dir, test_dir])

    make_dirs([test_dir_answer_answer, test_dir_headlines, test_dir_plagiarism, test_dir_postediting, test_dir_question_question])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.2-models.jar')])

    # split into separate files
    split(os.path.join(data_dir, 'SICK_train_big.txt'), train_dir)
    split(os.path.join(data_dir, 'SICK_trial_big.txt'), dev_dir)
    split(os.path.join(data_dir, 'SICK_test_answer-answer.txt'), test_dir_answer_answer)
    split(os.path.join(data_dir, 'SICK_test_headlines.txt'), test_dir_headlines)
    split(os.path.join(data_dir, 'SICK_test_plagiarism.txt'), test_dir_plagiarism)
    split(os.path.join(data_dir, 'SICK_test_postediting.txt'), test_dir_postediting)
    split(os.path.join(data_dir, 'SICK_test_question-question.txt'), test_dir_question_question)

    # parse sentences
    
    parse(train_dir, cp=classpath)
    parse(dev_dir, cp=classpath)

    parse(test_dir_answer_answer, cp=classpath)
    parse(test_dir_headlines, cp=classpath)
    parse(test_dir_plagiarism, cp=classpath)
    parse(test_dir_postediting, cp=classpath)
    parse(test_dir_question_question, cp=classpath)
    

    all_paths = []
    for fs in glob.glob(os.path.join(data_dir, '*/*.toks')):
        all_paths.append(fs)

    for fs in glob.glob(os.path.join(data_dir, '*/*/*.toks')):
        all_paths.append(fs)

    for fs in all_paths:
        print fs

    # get vocabulary
    build_vocab(all_paths, os.path.join(data_dir, 'vocab-cased.txt'),lowercase=False)

    build_word2Vector(os.path.join("../NLP-Tools", 'glove.840B.300d.txt'), data_dir, 'vocab-cased.txt')
