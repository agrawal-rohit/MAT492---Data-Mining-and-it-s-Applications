# Latent Semantic Analysis
import dask.array as da
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from os import listdir
from os.path import isfile, isdir, splitext, join
import random
import numpy as np
from sparsesvd import sparsesvd
import re
import scipy
import nltk
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
                
# Tokenizing
sentence_dict={}
pattern='.*?\.\s*[A-Z]'
p=re.compile(pattern)
for m in p.finditer(corpus):
    if m.start()!=0 and m.end()!=len(corpus):
            sentence_dict[len(sentence_dict)]=corpus[m.start()-1:m.end()-1]
    elif m.start()==0 and m.end()!=len(corpus):
        sentence_dict[len(sentence_dict)]=(corpus[m.start():m.end()-1])
    elif m.start==0 and m.end()==len(corpus):
        sentence_dict[len(sentence_dict)]=(corpus)
    else:
        sentence_dict[len(sentence_dict)]=(corpus[m.start()-1:m.end()])
sentence_dict[len(sentence_dict)]=corpus[m.end()-1:]

Word_dict={}
for sen in sentence_dict:
    words=sentence_dict[sen].split()
    for word in words:
        word=word.lower()
        if word not in Word_dict:
            Word_dict[word]=len(Word_dict)
            
# Compute TF-IDF matrix of sentence
M = np.zeros([len(Word_dict),len(sentence_dict)])
for sen in sentence_dict:
    words=sentence_dict[sen].split()
    words=[word.lower() for word in words]
    freqs=Counter(words)
    for key in freqs:
        M[Word_dict[key]][sen]=freqs[key]
        ##using just frequency of word in sentence as IDF is not available
        ##also one can weight each individual word based on if it is noun or verb etc.
        
# LSA
def heapify(L,index,k):
    while(2*index+1<k):
        t=1
        if(L[index][0]<L[2*index+1][0]):
            if 2*index+2>k or L[index][0]<L[2*index+2][0]:
                return L
            else:
                t=2
        else:
            if (t==1 and (2*index+2>=k) or L[2*index+1][0]<L[2*index+2][0]):
                t=1
            else:
                t=2

        temp=L[index]
        L[index]=L[2*index+t]
        L[2*index+t]=temp
        index=2*index+t
    return L        
        
def LSA(M,k):  ##will return top k sentences
    SM = scipy.sparse.csc_matrix(M) # convert to sparse CSC format
    u, s, vt = scipy.sparse.linalg.svds(M, k+10)
    ##SVD calculated at this stage, concept matrix vt, from now we can apply various approaches
    ##to filter out top k sentences.
    ##We are using OzSoy's approach
    ##Using Cross Method
    m,n=M.shape

    Avg=np.average(M,1)
    for i in range(0,m):
        for j in range(0,n):
            if M[i][j]<Avg[i]:
                M[i][j]=0
    Length=np.dot(s,vt)
    L=[]
    ##returning top k sentences
    for i in range(0,n):
        L.append(tuple([Length[i],i]))

    if k>=len(L):
        return L
    #building min heap

    count= int(k/2-1)

    while(count>=0):
        L=heapify(L,count,k)
        count-=1
    for i in range(k,len(L)):
        if L[0][0]<L[i][0]:
            L[0]=L[i]
            L=heapify(L,0,k)
    return L[:k]

L = LSA(M, 2)
L=sorted(L,key=lambda s : s[1])
print(L)

generated_summary = ""
for i in L:
    generated_summary += generated_summary + ' ' + str(sentence_dict[i[1]]).strip()
    
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
