
# coding: utf-8

# In[1]:


import os
import json
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import gensim.models.keyedvectors as word2vec
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split
import theano
theano.config.optimizer="None"
import pickle


# In[2]:


vec_x,vec_y = pickle.load(open("conversation.pickle","rb"))


# In[3]:


vec_x = np.array(vec_x,dtype = np.float64)
vec_y = np.array(vec_y,dtype = np.float64)

print(vec_x.shape)
print(vec_y.shape)


# In[4]:


x_train,x_test,y_train,y_test = train_test_split(vec_x,vec_y,test_size = 0.2,random_state = 1)


# In[5]:


model = Sequential()
model.add(LSTM(output_dim = 300,input_shape=x_train.shape[1:],return_sequences=True,init = 'glorot_normal',inner_init = 'glorot_normal',activation='sigmoid'))
model.add(LSTM(output_dim = 300,input_shape=x_train.shape[1:],return_sequences=True,init = 'glorot_normal',inner_init = 'glorot_normal',activation='sigmoid'))
model.add(LSTM(output_dim = 300,input_shape=x_train.shape[1:],return_sequences=True,init = 'glorot_normal',inner_init = 'glorot_normal',activation='sigmoid'))
model.add(LSTM(output_dim = 300,input_shape=x_train.shape[1:],return_sequences=True,init = 'glorot_normal',inner_init = 'glorot_normal',activation='sigmoid'))
model.compile(loss = 'cosine_proximity',optimizer = 'adam',metrics=['accuracy'])


# In[6]:


model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM500.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM1000.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM1500.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM2000.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM2500.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM3000.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM3500.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM4000.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM4500.h5')
model.fit(x_train,y_train,nb_epoch=500,validation_data = (x_test,y_test))
model.save('LSTM5000.h5')



# In[9]:


predictions = model.predict(x_test)
mod = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
[mod.most_similar([predictions[10][i]]) for i in range(15)]

