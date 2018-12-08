import os
import re
import pickle
import nltk
import numpy as np
import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

# Noun Part of Speech Tags used by NLTK
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

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

def merge_acronyms(s):
    """Merges all acronyms in a given sentence. For example M.I.T -> MIT"""
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.',''))
    return s

def clean_document(document):
    """Cleans document by removing unnecessary punctuation. It also removes
    any extra periods and merges acronyms to prevent the tokenizer from
    splitting a false sentence
    """
    # Remove all characters outside of Alpha Numeric
    # and some punctuation
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = document.replace('-', '')
    document = document.replace('...', '')
    document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

    # Remove Ancronymns M.I.T. -> MIT
    # to help with sentence tokenizing
    document = merge_acronyms(document)

    # Remove extra whitespace
    document = ' '.join(document.split())
    return document

def remove_stop_words(document):
    """Returns document without stop words"""
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def rank_sentences(doc, doc_matrix, feature_names, top_n=3):
    """Returns top_n sentences. Theses sentences are then used as summary
    of document.
    input
    ------------
    doc : a document as type str
    doc_matrix : a dense tf-idf matrix calculated with Scikits TfidfTransformer
    feature_names : a list of all features, the index is used to look up
                    tf-idf scores in the doc_matrix
    top_n : number of sentences to return
    """
    sents = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS]
                  for sent in sentences]
    tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                   for w in sent if w.lower() in feature_names]
                 for sent in sentences]

    # Calculate Sentence Values
    doc_val = sum(doc_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

    # Apply Position Weights
    ranked_sents = [sent*(i/len(sent_values))
                    for i, sent in enumerate(sent_values)]

    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)

    return ranked_sents[:top_n]

import tarfile
import numpy as np 

if __name__ == '__main__':
    # Load corpus data used to train the TF-IDF Transformer
    wikicorpus = []
    tar = tarfile.open("../Data/raw.en.tgz", "r:gz")
    counter = 0
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if counter is 10:
            break
        if f:
            counter+=1
            wikicorpus.append(str(f.read()))
            
    # Load the document you wish to summarize
    document = corpus

    cleaned_document = clean_document(document)
    doc = remove_stop_words(cleaned_document)

    # Merge corpus data and new document data
    data = [' '.join(wikicorpus)] 
    train_data = set(data + [doc])

    # Fit and Transform the term frequencies into a vector
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(train_data)
    freq_term_matrix = count_vect.transform(train_data)
    feature_names = count_vect.get_feature_names()

    # Fit and Transform the TfidfTransformer
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)

    # Get the dense tf-idf matrix for the document
    story_freq_term_matrix = count_vect.transform([doc])
    story_tfidf_matrix = tfidf.transform(story_freq_term_matrix)
    story_dense = story_tfidf_matrix.todense()
    doc_matrix = story_dense.tolist()[0]

    # Get Top Ranking Sentences and join them as a summary
    top_sents = rank_sentences(doc, doc_matrix, feature_names, 2)
    summary = '.'.join([cleaned_document.split('.')[i]
                        for i in [pair[0] for pair in top_sents]])
    summary = ' '.join(summary.split())
    print(summary)

# Evaluation
rouge = RougeCalculator(stopwords=True, lang="en")
bleu = BLEUCalculator()

rouge_1_scores = []
rouge_1_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[0],
            n=1))

rouge_1_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[1],
            n=1))

rouge_1_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[2],
            n=1))

rouge_1_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[3],
            n=1))

rouge_1_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[4],
            n=1))
mean_rouge_1_score = np.mean(rouge_1_scores)

rouge_2_scores = []
rouge_2_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[0],
            n=2))

rouge_2_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[1],
            n=2))

rouge_2_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[2],
            n=2))

rouge_2_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[3],
            n=2))

rouge_2_scores.append(rouge.rouge_n(
            summary=summary,
            references=summaries[4],
            n=2))
mean_rouge_2_score = np.mean(rouge_2_scores)

bleu_scores = []
bleu_scores.append(bleu.bleu(summary=summary,
                  references=summaries[0]))
bleu_scores.append(bleu.bleu(summary=summary,
                  references=summaries[1]))
bleu_scores.append(bleu.bleu(summary=summary,
                  references=summaries[2]))
bleu_scores.append(bleu.bleu(summary=summary,
                  references=summaries[3]))
bleu_scores.append(bleu.bleu(summary=summary,
                  references=summaries[4]))
mean_bleu_scores = np.mean(bleu_scores)
