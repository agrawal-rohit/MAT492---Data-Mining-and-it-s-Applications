import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt') # one time execution
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from os import listdir
from os.path import isfile, isdir, splitext, join
import random
import re
import numpy as np
import nltk
import heapq  
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator

# Read corpus
corpus_root = "../Data/Opinosis/topics/"
corpus_files = [s for s in listdir(corpus_root) if splitext(s)[1] == ".data"]
corpus_file = random.choice(corpus_files)      # Read random file
current_topic = splitext(splitext(corpus_file)[0])[0]
with open(join(corpus_root, corpus_file), 'r') as corpus_file_handle:
    corpus = corpus_file_handle.read()
    
# Read ground truth summaries
summaries = []
summaries_root = "../Data/Opinosis/summaries-gold/"
summary_files = [s for s in listdir(join(summaries_root, current_topic)) if splitext(s)[1] == ".gold"]
for summary_file in summary_files:
    with open(join(join(summaries_root, current_topic), summary_file), 'r') as summary_file_handle:
                summaries.append(summary_file_handle.read())

# split the the text in the articles into sentences
sentences =  nltk.sent_tokenize(corpus)  

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]

stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sen):
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new

# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

# Extract word vectors
word_embeddings = {}
f = open('../Data/glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)
  
# Similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
      
import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Specify number of sentences to form the summary
sn = 2
generated_summary = ''
# Generate summary
for i in range(sn):
  generated_summary = ''.join(ranked_sentences[i][1])
  
# Evaluation
rouge = RougeCalculator(stopwords=True, lang="en")
bleu = BLEUCalculator()

rouge_1_scores = []
rouge_1_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[0],
            n=1))

rouge_1_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[1],
            n=1))

rouge_1_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[2],
            n=1))

rouge_1_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[3],
            n=1))

rouge_1_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[4],
            n=1))
mean_rouge_1_score = np.mean(rouge_1_scores)

rouge_2_scores = []
rouge_2_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[0],
            n=2))

rouge_2_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[1],
            n=2))

rouge_2_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[2],
            n=2))

rouge_2_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[3],
            n=2))

rouge_2_scores.append(rouge.rouge_n(
            summary=generated_summary,
            references=summaries[4],
            n=2))
mean_rouge_2_score = np.mean(rouge_2_scores)

bleu_scores = []
bleu_scores.append(bleu.bleu(summary=generated_summary,
                  references=summaries[0]))
bleu_scores.append(bleu.bleu(summary=generated_summary,
                  references=summaries[1]))
bleu_scores.append(bleu.bleu(summary=generated_summary,
                  references=summaries[2]))
bleu_scores.append(bleu.bleu(summary=generated_summary,
                  references=summaries[3]))
bleu_scores.append(bleu.bleu(summary=generated_summary,
                  references=summaries[4]))
mean_bleu_scores = np.mean(bleu_scores)
