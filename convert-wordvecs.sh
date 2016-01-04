#!/bin/bash
glove_dir='data/glove'
glove_pre='glove.840B'
glove_dim='300d'

th convert-wordvecs.lua /Users/peng/Develops/NLP-Tools/$glove_pre.$glove_dim.txt \
	$glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th