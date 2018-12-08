# Term Frequency Summarization

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

# PREPROCESSING
# -------------
# Removing Square Brackets and Extra Spaces
corpus = re.sub(r'\[[0-9]*\]', ' ', corpus)  
corpus = re.sub(r'\s+', ' ', corpus)  

# Removing special characters and digits
formatted_corpus = re.sub('[^a-zA-Z]', ' ', corpus )  
formatted_corpus = re.sub(r'\s+', ' ', corpus) 
sentence_list = nltk.sent_tokenize(corpus)  
stopwords = stopwords.words('english')

# Calculate Word Frequencies
word_frequencies = {}  
for word in nltk.word_tokenize(formatted_corpus):  
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
            
maximum_frequncy = max(word_frequencies.values())
for word in word_frequencies.keys():  
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    
sentence_scores = {}  
for sent in sentence_list:  
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

# Getting the Summary
summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
generated_summary = ' '.join(summary_sentences)  
print(generated_summary)  

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
