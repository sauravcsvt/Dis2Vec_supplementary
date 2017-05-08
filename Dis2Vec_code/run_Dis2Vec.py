#!/usr/bin/python
# -*- coding: utf-8 -*-
# :LICENSE: MIT

__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"

import os
os.system("unzip Dis2Vec.zip -d ./")
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("./Dis2Vec/")
import cPickle as pickle
from gensim.models import Word2Vec
import time
import logging
import argparse
logging.basicConfig()

class Dis2Vec(object):

    def __init__(self, **kwargs):
        self.params = kwargs

    def run_Dis2Vec(self):

        start_time = time.time()
        model_HM = Word2Vec(self.params["sent"], size=self.params["dim"], window=self.params["win"], 
                            min_count=self.params["min_cnt"], sample=self.params["sample"], workers=100,
                            sg=1, hs=0, negative=self.params["neg"], iter=self.params["iter"], 
                            sampling_param=self.params["spm"], objective_param=self.params["opm"], 
                            smoothing=self.params["smoothing"], vocab_file=self.params["vocab"])
        end_time = time.time()
        print "Total time taken is: " + str((end_time - start_time) / 3600.) + " hours"
        out_folder = './output/'
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        if model_HM.sample == 0:
            model_HM.save(out_folder + 'model_Dis2Vec.word2vec')
        else:
            model_HM.save(out_folder + 'model_Dis2Vec_w_sample.word2vec')

def parse_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("-ic", "--inputcorpus", type = str, required = True, help = "Input corpus which should be a list of sentences as input where each sentence is a list of tokens. file should be in .pkl format")
    ap.add_argument("-v", "--domainvocab", type = str, required = True, help = "Domain-specific vocabulary. file should be in .pkl format")
    return ap.parse_args()

def main():

    _arg = parse_args()
    sentences_corpus = pickle.load(open(_arg.inputcorpus, "r")) # Input corpus (list of sentences as input where each sentence is a list of tokens. file should be in .pkl format).
    domain_vocab_file = _arg.domainvocab # domain-specific vocabulary. file should be in .pkl format
    kwargs = {"sent": sentences_corpus, "vocab": domain_vocab_file, 
              "dim": 300, "win": 15, "min_cnt": 5, "neg": 1, "iter": 1, 
              "spm": 0.3, "opm": 0.7, "smoothing": 1.0, "sample": 1e-05}
    Dis2Vec(**kwargs).run_Dis2Vec()

if __name__ == "__main__":
    main()
