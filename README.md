# Semantic Text Similarity 

This folder contains the codes and the tools to the run the semantic text similarity calculator.

------------------------------------------------------
The code files are:
- main.py : main file of the task that gets the input arguments, run the similarity calculator, and save the result file.
- data_loader.py : this file reads the input file in txt format and extracts the two sentences from each line of file.
- sentencesimilarity.py : it calculate the similarity between the sentence couples in the input dataset with respect to sentence embedder of choice.

The zip files are:
- infersent.zip : this file containts the code for infersent tool which is downloaded from https://github.com/facebookresearch/InferSent github repository.
- doc2vec_pretrained: this file contains the doc2vec pretrained model which is downloaded from https://github.com/jhlau/doc2vec github repository with name of Associated Press News DBOW (0.6GB).

The other folder:
results: this folder contains the results of the sentence similarity code for each dataset and pretrained model

How to run:

1. install the requirements:
$ pip install -r requirements.txt

2. unzip infersent.zip 

3. unzip doc2vec_pretrained.zip

4. download the infersent pretrained model (Glove) by running these commands (they should be run at the location of main.py file):
$ mkdir encoder
$ curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
$ mkdir GloVe
$ curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
$ unzip GloVe/glove.840B.300d.zip -d GloVe/

5. run the main.py with input dataset file:
$ python main.py --SE <model option> --outputfile <name of outputfile>

The model options are SBERT, Doc2Vec, UniversalSentenceEncoder, fastText, InferSent (exactly as it is written here). The default option is SBERT as the best performance model between them.
The default option of outputfile name is in SYSTEM_OUT.<dataset name>.txt like SYSYEM_OUT.answer-answer.txt

--------------
Example:
$ python main.py sts2016-english-with-gs-v1.0/STS2016.input.headlines.txt --SE SBERT --outputfile results_sbert/SYSTEM_OUT.headlines.txt
