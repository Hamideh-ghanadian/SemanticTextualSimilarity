"""
This file contains the main part of the automata-based data generator.
Input parameters are as follows:
dataset: the location of the dataset
log_template_list: the location of the csv file of log template ID to log template text
--split_rate: the splitting rate fro train set, default is 80 percent
--max_input_len', help="the maximum length of input sequence of the language model, default is 75
--epoch_num: the number of epochs for training, default is 10
--batch_size: the size of batch for training, default is 64
--tb_num: the number of transformer block in stack inside the model, default is 1
--embed_len: the length of embedding vector for each log template, default is 768
"""

import argparse
import pickle 
import os
from data_loader import Dataset
from sentencesimilarity import SentenceSimilarity

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('inputfile', help="the location of the input file in txt. file",
                        type=str)
    parser.add_argument('--SE', help="the name of the embedder of choice",
                        type=str, default="SBERT")
    parser.add_argument('--outputfile', help="the name of the output file in txt. file",
                        type=str, default="")
    args = parser.parse_args()

    # load the dataset from the input file location
    dataset = Dataset(args.inputfile)

    # compute the sentence similarities
    ss = SentenceSimilarity(dataset, args.SE, args.outputfile)

    # save the results in a txt file in format of SYSTEM_OUT.<dataset name>.txt
    ss.save_results()
  
