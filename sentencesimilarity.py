from csv import reader
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import nltk
import torch
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import sister
from infersent.models import InferSent

# download nltk for InferSent
nltk.download('punkt')

class SentenceSimilarity:
    def __init__(self, dataset, SE, outputfile):
        # initialized the dataset object
        self.dataset = dataset

        # initialized the name of the output file
        self.outputfile_name = outputfile

        # initialized the choice of sentence embedder model
        self.SE = SE

        # calculate the similarity of the dataset sentences
        self.results = self.calculate_similairy()

    # calculate the similarity of the dataset sentences
    def calculate_similairy(self):
      # stores results as a list
      results = []

      # if the chosen sentence embedder is SBERT
      if self.SE == "SBERT":

        sentences1 = []  # stores all the first sentences
        sentences2 = []  # stores all the second sentences

        # load the pre-trained model 
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        for data in self.dataset.dataset:
          sentences1.append(data[0])
          sentences2.append(data[1])

        #Compute the embeddings for the lists of sentneces 
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)
          
        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        # stores the results as a floating point between 0 to 5.
        results = [float(abs(cosine_scores[i][i]*5)) for i in range(len(cosine_scores))]

      elif self.SE =="Doc2Vec":
        
        # create the Doc2Vec model
        model = gensim.models.doc2vec.Doc2Vec(vector_size=30, min_count=2, epochs=80)

        # load a pretain model of Doc2Vec
        model = Doc2Vec.load("/content/doc2vec-pretrained/doc2vec.bin")
        
        # compute similarity for each couple of sentences
        for d in self.dataset.dataset:
          # tokenize the sentences and infer the embedding
          sentence1 = model.infer_vector(word_tokenize(d[0]))
          sentence2 = model.infer_vector(word_tokenize(d[1]))
          # compute cosine scores
          cosine_scores = util.cos_sim(sentence1, sentence2)[0][0]
          # store the results as a floating point between 0 to 5.
          results.append(float(abs(cosine_scores*5)))

      elif self.SE =="UniversalSentenceEncoder":

        sentences1 = []  # stores all the first sentences
        sentences2 = []  # stores all the second sentences

        # load the pretained Universal Sentence Encoder
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
        model = hub.load(module_url)

        # compute similarity for each couple of sentences
        for data in self.dataset.dataset:
          # infer vector embedding
          sentence1 = model([data[0]])
          sentence2 = model([data[1]])

          # compute cosine similarity
          m = tf.keras.metrics.CosineSimilarity(axis=1)
          m.update_state(sentence1,sentence2)

          # 
          results.append(float(abs(m.result().numpy()*5)))

      elif self.SE == "fastText":
        embeddings1 = [] # stores embeddings of the first sentences
        embeddings2 = [] # stores embeddings of the second sentences

        # set up the model to pre-trained model of MeanEmbedding for English
        model = sister.MeanEmbedding(lang="en")

        for data in self.dataset.dataset:
          # infer vector embedding
          sentence1=model(data[0])
          sentence2=model(data[1])

          # store the vector embedding
          embeddings1.append(sentence1)
          embeddings2.append(sentence2)
          
        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        # store the results as a floating point between 0 to 5.
        results = [float(abs(cosine_scores[i][i]*5)) for i in range(len(cosine_scores))]

      elif self.SE == "InferSent":

        sentences1 = []  # stores all the first sentences
        sentences2 = []  # stores all the second sentences

        # set up the pre-trained model
        model_version = 1 # use GloVe embeddings
        MODEL_PATH = "encoder/infersent%s.pkl" % model_version
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))

        # Keep it on CPU or put it on GPU
        use_cuda = False
        model = model.cuda() if use_cuda else model

        # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
        W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
        model.set_w2v_path(W2V_PATH)

        # Load embeddings of K most frequent words
        model.build_vocab_k_words(K=100000)

        for data in self.dataset.dataset:
          # infer vector embedding
          sentences1.append(data[0])
          sentences2.append(data[1])

        #Compute embedding for both lists  
        embeddings1 = model.encode(sentences1, bsize=128, tokenize=False, verbose=True)
        embeddings2 = model.encode(sentences2, bsize=128, tokenize=False, verbose=True)
          
        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        # store the results as a floating point between 0 to 5.
        results = [float(abs(cosine_scores[i][i]*5)) for i in range(len(cosine_scores))]

      # if the input SE model name is unknown 
      else:
        print("unknown embedder!")

      #print(results)
      return results
      
    # save the results results in a txt file
    def save_results(self):

      # if the name of output file is not specified in as input argument
      if self.outputfile_name == "":
        self.outputfile_name = "SYSTEM_OUT."+ self.dataset.name + ".txt"

      # if the name of output file is specified in as input argument
      with open(self.outputfile_name, 'w') as filehandle:
        for listitem in self.results:
          filehandle.write('%s\n' % listitem)

      return