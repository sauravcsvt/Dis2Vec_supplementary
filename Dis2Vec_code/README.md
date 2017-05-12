usage: run_Dis2Vec.py [-h] -ic INPUTCORPUS -v DOMAINVOCAB -d DIM -w WINDOW -n
                      NEGATIVE -spm SAMPLINGPARAMETER -opm OBJECTIVEPARAMETER
                      -sm SMOOTHING

optional arguments:
  -h, --help            show this help message and exit
  -ic INPUTCORPUS, --inputcorpus INPUTCORPUS
                        Input corpus which should be a list of sentences as
                        input where each sentence is a list of tokens. file
                        should be in .pkl format
  -v DOMAINVOCAB, --domainvocab DOMAINVOCAB
                        Domain-specific vocabulary. file should be in .pkl
                        format
  -d DIM, --dim DIM     Dimension of word embeddings (300, 600)
  -w WINDOW, --window WINDOW
                        Word window (5, 10, 15)
  -n NEGATIVE, --negative NEGATIVE
                        Number of negative samples (1, 5, 15)
  -spm SAMPLINGPARAMETER, --samplingparameter SAMPLINGPARAMETER
                        Sampling parameter (0.3, 0.5, 0.7)
  -opm OBJECTIVEPARAMETER, --objectiveparameter OBJECTIVEPARAMETER
                        Objective selection parameter (0.3, 0.5, 0.7)
  -sm SMOOTHING, --smoothing SMOOTHING
                        smoothing parameter (0.75, 1.0)
