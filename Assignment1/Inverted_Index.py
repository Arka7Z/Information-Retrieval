
# coding: utf-8

# In[7]:


import os
import nltk
from nltk.stem import WordNetLemmatizer
import io
path=os.getcwd()
porter=nltk.PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
postings=dict()
i=10
for file in os.listdir(path+"/alldocs"):
    f = io.open(os.path.join(path+"/alldocs", file), 'r', encoding='utf-8')
    file_content=f.read()
    tokens = nltk.word_tokenize(file_content)
    wordset=set(tokens)
    #wordset=set([wordnet_lemmatizer.lemmatize(token) for token in tokens])
    for word in wordset:
        if word in postings:
            postings[word].append(file)
        else:
            postings[word]=[file]



# In[8]:


from functools import reduce 
output_file=open("output.txt","w")
with open('query.txt') as fp:
    for line in fp:
        query=line.split()
        if(len(query)>=2):
            query_index=query[0]
            query=query[1:]
            #query=[wordnet_lemmatizer.lemmatize(query_word) for query_word in query]
            doc_list=[]
            for term in query:
                if term in postings:
                    doc_list.append(postings[term])
            doc_list=list(reduce(set.intersection, [set(item) for item in doc_list ]))
            print(query_index)
            print(doc_list)
            counter=0
            for doc in doc_list:
                output_file.write(query_index+" "+doc)
                output_file.write("\n")
                counter+=1
                if(counter==50):
                    break
            #print query
fp.close()
output_file.close()


# In[11]:


print(postings['pearl'])


# In[ ]:


p

