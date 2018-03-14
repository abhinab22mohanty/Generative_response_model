
# coding: utf-8

# In[1]:


import os
import json
import nltk
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import gensim.models.keyedvectors as word2vec
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from gensim import corpora,models,similarities
import pickle


# In[2]:


model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[3]:


file = open('conversations.json')
data = json.load(file)
cor = data["conversations"]


# In[7]:


x = []
y = []

#Appending in question answer format
for i in range(len(cor)):
    for j in range(len(cor)):
        if j<len(cor[i])-1:
            x.append(cor[i][j])
            y.append(cor[i][j+1])
            
print(x[1])
print(y[1])


# In[8]:


tok_x = []
tok_y = []

for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))
    
print(tok_x[1])
print(tok_y[1])


# In[11]:


sentend = np.ones((300,),dtype = np.float32)
print(sentend)


# In[13]:


#Mapping each word in our corpus to the word2vec in W2V model
vec_x = []
for word in tok_x:
    wordvec = [model[w] for w in word if w in model.vocab]
    vec_x.append(wordvec)
    
print(wordvec)


# In[14]:


vec_y = []
for word in tok_y:
    wordvec = [model[w] for w in word if w in model.vocab]
    vec_y.append(wordvec)


# In[15]:


#Clipping sentences greater than 14... Appending 1 vectors for 15th....eg. Hi->14vec->sentend

for tok_wor in vec_x:
    tok_wor[14:] = []
    tok_wor.append(sentend)


# In[16]:


for tok_wor in vec_x:
    if len(tok_wor)<15:
        for i in range(15-len(tok_wor)):
            tok_wor.append(sentend)
            

            
for tok_wor in vec_y:
    tok_wor[14:] = []
    tok_wor.append(sentend)
    
    

for tok_wor in vec_y:
    if len(tok_wor)<15:
        for i in range(15-len(tok_wor)):
            tok_wor.append(sentend)


# In[17]:


pickle.dump( [vec_x,vec_y], open( "conversation.pickle", "wb" ) )

