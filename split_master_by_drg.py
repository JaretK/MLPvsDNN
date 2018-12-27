#!/usr/bin/env python -W ignore
from __future__ import print_function

import pandas as pd 
from os.path import abspath
from os.path import exists
from os import mkdir
from absl import app
from absl import flags

FLAGS = flags.FLAGS
# delcare flags
flags.DEFINE_string("master", None, "Master SPARCS file to split by drg")
flags.DEFINE_string("output", None, "Output directory in which to save output files")

# required flags
flags.mark_flag_as_required("master")
flags.mark_flag_as_required("output")


def process_master_to_drg(master_df, output):
    all_drg = master_df['apr_drg_code']
    uniq = all_drg.unique()
    for x in uniq:
        x_vals = master_df.loc[master_df['apr_drg_code'] == x]
        filename = '%s/%s.csv' % (output, x)
        x_vals.to_csv(filename, index=False)
        print("Saved %d samples in %s" % (x_vals.shape[0], filename))
        del x_vals
        

def main(argv):
    df = FLAGS.master
    abs_df = abspath(df)
    abs_output = abspath(FLAGS.output)
    if not exists(abs_output):
        mkdir(abs_output)
    print("Processing \"%s\"" % abs_df)
    csv = pd.read_csv(abs_df)
    processed = process_master_to_drg(csv, abs_output)
    return 


if __name__ == "__main__":
    app.run(main)

