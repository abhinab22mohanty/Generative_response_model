
# coding: utf-8

# In[1]:


import os
from scipy import spatial
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import gensim.models.keyedvectors as word2vec
import nltk
from keras.models import load_model


# In[2]:


import theano
theano.config.optimizer="None"


# In[3]:


model = load_model('LSTM5000.h5')
mod = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

while(True):
    x = raw_input("Enter message:")
    sentend = np.ones((300,),dtype = np.float32)
    
    sent = nltk.word_tokenize(x.lower())
    sent_vec = [mod[w] for w in sent if w in mod.vocab]
    
    sent_vec[14:] = []
    sent_vec.append(sentend)
    
    #Similar Clipping as train
    if(len(sent_vec)<15):
        for i in range(15-len(sent_vec)):
            sent_vec.append(sentend)
        sentvec = np.array([sent_vec])
        
    predictions = model.predict(sentvec)
    outputlist = [mod.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    
    output = ' '.join(outputlist)
    print(output)

