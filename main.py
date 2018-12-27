#!/usr/bin/env python3 -W ignore

###############################
#
# Neural Network Prediction of cost
# after orthopaedic and neurosurgical
# procedures. 
# 
# Please run this under a jupyter environment.
# Tested using python 3.7 on Mac OS Sierra
# using Microsoft visual studio code jupyter 
# extension
# 
# This code is subject to change 
# (see commit log)
#
# Copyright 2018 Jaret M. Karnuta under 
# licenses provided herein 


#%% Import statements and load datasets/objects
# Futures
from __future__ import division
from __future__ import print_function

# Google abseil library
from absl import app
from absl import flags

# other imports
import os
import pandas as pd 
from logging import logging

# absl-py flags
FLAGS = flags.FLAGS
# delcare flags
flags.DEFINE_string("df", None, "SPARCS file used to run ANN")
flags.DEFINE_string("output", None, "Output directory in which to save output files")

# required flags
flags.mark_flag_as_required("df")
flags.mark_flag_as_required("output")

#%%
def main(argv):
    output = FLAGS.output
    df = pd.read_csv(FLAGS.df, low_memory = False)
    print('Loaded')
    logging.basicConfig(filename='logger.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    ########################
    # Run user-defined model
    

if __name__ == "__main__":
    app.run(main)


