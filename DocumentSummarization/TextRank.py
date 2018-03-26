
# coding: utf-8

# In[ ]:


from __future__ import division
import os
import nltk
import time
import pickle
import io
import math
import numpy as np
from nltk.stem import WordNetLemmatizer
from functools import reduce
from bs4 import BeautifulSoup
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from scipy import spatial

word_to_id_map=dict()
id_to_word=dict()
word_to_doc_freq=dict()
sentence_to_id=dict()
id_to_sentence=dict()
sentence_count=0
word_count=0
total_docs=0
sentence_tf_idf_vec=dict()
wordnet_lemmatizer = WordNetLemmatizer()
path=os.getcwd()


# In[ ]:


def init():
    global word_to_id_map
    global id_to_word
    global word_to_doc_freq
    global sentence_to_id
    global id_to_sentence
    global sentence_count
    global word_count
    global total_docs
    global sentence_tf_idf_vec
    word_to_id_map=dict()
    id_to_word=dict()
    word_to_doc_freq=dict()
    sentence_to_id=dict()
    id_to_sentence=dict()
    sentence_count=0
    word_count=0
    total_docs=0
    sentence_tf_idf_vec=dict()


# In[ ]:


# Module to parse the file and get the sentences

def parse_file(topic_num):
    global word_to_id_map
    global id_to_word
    global word_to_doc_freq
    global sentence_to_id
    global id_to_sentence
    global sentence_count
    global word_count
    global total_docs
    global sentence_tf_idf_vec
    
    for file in os.listdir(path+"/Assignement2_IR/Topic"+str(topic_num)):
        total_docs+=1
        f = io.open(os.path.join(path+"/Assignement2_IR/Topic"+str(topic_num), file), 'r', encoding='utf-8')
        file_content=f.read()
        soup = BeautifulSoup(file_content, "lxml")
        text_group = soup.get_text()
        text_group = ' '.join(text_group.strip().split('\n'))
        sentences=nltk.sent_tokenize(text_group)
        doc_word_set=set()
        
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            tokens=[token.lower() for token in tokens]
            wordset=[wordnet_lemmatizer.lemmatize(token) for token in tokens]
            sentence_to_id[sentence]=sentence_count
            id_to_sentence[sentence_count]=sentence
            for word in wordset:
                doc_word_set.add(word)
                if word in word_to_id_map:
                    ("")
                else:
                    word_to_id_map[word]=word_count
                    id_to_word[word_count]=word
                    word_count+=1
            sentence_count+=1
        
        for dist_word in doc_word_set:
            if dist_word in word_to_doc_freq:
                word_to_doc_freq[dist_word]+=1
            else:
                word_to_doc_freq[dist_word]=1


# In[ ]:


# CREATING TF-IDF representation of the sentences

def get_tf_idf_vec():
    global word_to_id_map
    global id_to_word
    global word_to_doc_freq
    global sentence_to_id
    global id_to_sentence
    global sentence_count
    global word_count
    global total_docs
    global sentence_tf_idf_vec

    for sentence in sentence_to_id:
        tokens = nltk.word_tokenize(sentence)
        tokens=[token.lower() for token in tokens]
        wordset=[wordnet_lemmatizer.lemmatize(token) for token in tokens]
        sentence_tf_idf_vec[sentence]=[0.0]*word_count
        for word in wordset:
            sentence_tf_idf_vec[sentence][word_to_id_map[word]]+=1
        wordset=list(set(wordset))
        for dist_word in wordset:
            sentence_tf_idf_vec[sentence][word_to_id_map[dist_word]]*=math.log(total_docs/word_to_doc_freq[dist_word])/math.log(2)


# In[ ]:


# Returns cosine similarity between the vector representation of the two sentences

def get_cosine_similarity(sentence1,sentence2):
    global word_to_id_map
    global id_to_word
    global word_to_doc_freq
    global sentence_to_id
    global id_to_sentence
    global sentence_count
    global word_count
    global total_docs
    global sentence_tf_idf_vec
    sim=1 - spatial.distance.cosine(sentence_tf_idf_vec[sentence1], sentence_tf_idf_vec[sentence2])
    return sim


# In[ ]:


# Constructs the similarity graph between sentences with threshold passed as param

def construct_graph(threshold):
    global word_to_id_map
    global id_to_word
    global word_to_doc_freq
    global sentence_to_id
    global id_to_sentence
    global sentence_count
    global word_count
    global total_docs
    global sentence_tf_idf_vec
    graph=dict()
    Degree=[0]*sentence_count
    
    for i in range(sentence_count):
        graph[i]=[0.0]*sentence_count
        for j in range (sentence_count):
            graph[i][j]=get_cosine_similarity(id_to_sentence[i],id_to_sentence[j])
            if(graph[i][j]<threshold):
                graph[i][j]=0
            else:
                graph[i][j]=1
                Degree[i]+=1
    
    for i in range(sentence_count):
        for j in range (sentence_count):
            if(Degree[i]!=0):
                graph[i][j]/=Degree[i]
                
    return graph,Degree


# In[ ]:


# Power method to find the left eigen vector of the transition kernel

def power_method(graph,damping_factor,tolerance):
    d=damping_factor
    global sentence_count
    
    tmp=list()
    for key,value in graph.items():
        tmp.append(value)
    graph=np.asarray(tmp)
    p_old=[1.0/sentence_count]*sentence_count
    U=[p_old]*len(p_old)
    U=np.asarray(U)
    p_old=np.asarray(p_old).T
    iter_count=0
    while(True):
        iter_count+=1
        p=graph.T.dot(p_old)
        p=(d*U+(1-d)*graph).T.dot(p_old)                     
        if (  (np.sqrt(np.sum((p_old-p)**2))) <tolerance):
            break
        p_old=p
    return p


# In[ ]:


# Selects sentences on the basis of TextRank algorithm and writes them to a file

def write_summary_to_file(topic_num,threshold):
    global word_to_id_map
    global id_to_word
    global word_to_doc_freq
    global sentence_to_id
    global id_to_sentence
    global sentence_count
    global word_count
    global total_docs
    global sentence_tf_idf_vec
    
    init()
    parse_file(topic_num)
    get_tf_idf_vec()
    graph,Degree=construct_graph(threshold)
    
    p=power_method(graph,0.15,1e-5)
    words_added=0
    summary=""
    p=np.argsort(p)
    p=p[::-1]
    for highest_degree_id in p:
        sentence=id_to_sentence[highest_degree_id]
        summary+=sentence
        tokens = nltk.word_tokenize(sentence)
        words_added+=len(tokens)
        if(words_added>250):
            break
    
    with open("TextRank_Summary_"+str(topic_num)+"_Threshold_"+str(threshold)+".txt",'a') as file:
        file.write(summary)
    file.close()   


# In[ ]:


threshold_list=[0.1,0.2,0.3]
for topic in range(1,6):
    for thresh in threshold_list:
        write_summary_to_file(topic,thresh)


# In[ ]:




