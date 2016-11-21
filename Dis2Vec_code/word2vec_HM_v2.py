#!/usr/bin/python
# -*- coding: utf-8 -*-
# :LICENSE: MIT

__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"

# import os
# import glob
# import multiprocessing
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# import json
# import re
import cPickle as pickle
# from embers import utils
from gensim.models import Word2Vec
import time

import logging
logging.basicConfig()


def perform_word2vec(sent_list, vec_dim, win_size, count_min, SG_negative, iter_corpus, spm, opm, smoothing):

    print len(sent_list), vec_dim, win_size, count_min, SG_negative, iter_corpus, spm, opm, smoothing
    start_time = time.time()
    model_HM = Word2Vec(sent_list, size=vec_dim, window=win_size, 
                        min_count=count_min, sample=1e-05, workers=100,
                        sg=1, hs=0, negative=SG_negative, iter=iter_corpus, sampling_param=spm, objective_param=opm, smoothing=smoothing)
    print model_HM
    end_time = time.time()
    print (end_time - start_time) / 3600., "hours"
    out_folder = '../english_articles/model_v2/model_all_vocab_full_VSGNS/'
    if model_HM.sample == 0:
        model_HM.save(out_folder + 'model_HM_all_vocab_' + str(vec_dim) + "_" + str(win_size) + "_" + str(count_min) + "_" + str(SG_negative) + "_" + str(spm) + "_" + str(opm) + "_" + str(smoothing) + "_" + str(iter_corpus) + '.word2vec')
    else:
        model_HM.save(out_folder + 'model_HM_all_vocab_' + str(vec_dim) + "_" + str(win_size) + "_" + str(count_min) + "_" + str(SG_negative) + "_" + str(spm) + "_" + str(opm) + "_" + str(smoothing) + "_" + str(iter_corpus) + 'w_sample.word2vec')


def main():
    
    sentences_word2vec = pickle.load(open("../english_articles/model_v2/sentence_w_phrases_input_word2vec.pkl", "r"))
    param_list = {"dim": 600, "win": [15], "min_cnt": 5, "neg": [1, 5, 15], "iter": 1, "spm": [0.3, 0.5, 0.7], "opm": [0.3, 0.5, 0.7], "smoothing": [0.75, 1.0]}
    for param_win in param_list['win']:
        for param_neg in param_list['neg']:
            for param_spm in param_list['spm']:
                for param_opm in param_list['opm']:
                    for param_smooth in param_list["smoothing"]:
                        perform_word2vec(sentences_word2vec, param_list["dim"], param_win, param_list["min_cnt"], 
                                         param_neg, param_list["iter"], param_spm, param_opm, param_smooth)

if __name__ == "__main__":
    main()
