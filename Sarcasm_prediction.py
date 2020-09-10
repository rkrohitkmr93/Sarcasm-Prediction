#!/usr/bin/env python
# coding: utf-8

# In[344]:


import pandas as pd
import numpy as np


# In[347]:


#loading the train data
df_train = pd.read_json("train.json", lines = True)
df_train.head()


# In[351]:


#To know the percentage of both sarcasm and non-sarcasm headlines

import plotly as py
from plotly import graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

#before cleaning the data
# Making pie chart to compare the numbers of sarcastic and not-sarcastic headlines
labels = ['Sarcastic', 'Not Sarcastic']
count_sarcastic = len(df_train[df_train['is_sarcastic']==1])
count_notsarcastic = len(df_train[df_train['is_sarcastic']==0])
values = [count_sarcastic, count_notsarcastic]
# values = [20,50]

trace = go.Pie(labels=labels,
               values=values,
               textfont=dict(size=19, color='#000000'),
               marker=dict(
                   colors=['#FFFF00', '#2424FF'] 
               )
              )

layout = go.Layout(title = '<b>Sarcastic vs Not Sarcastic</b>')
data = [trace]
fig = go.Figure(data=data, layout=layout)

iplot(fig)


# In[352]:


#load the test data
df_test = pd.read_json("test.json", lines = True)
df_test.head()


# In[353]:


df3 = pd.DataFrame(df_train, columns = ['article_link', 'headline'])
df3.head()


# In[354]:


df3.info()


# In[355]:


#merge both train and test data columns(article_link, headline) 
#so that preprocessing will have to do only one time

frame = [df3, df_test]
df = pd.concat(frame, ignore_index=True)
df.head()


# In[289]:


df.info()


# In[356]:


#check the headline column
for i,headline in enumerate (df['headline'], 1):
    if i > 20:
        break
    else:
        print(i, headline)


# In[357]:


#To remove punctuations, digits and numbers
#Text cleaning

import string
from string import digits, punctuation

hl_cleansed = []
for hl in df['headline']:
    #Remove punctuations
    clean = hl.translate(str.maketrans('', '', punctuation))
    #Remove digits/numbers
    clean = clean.translate(str.maketrans('', '', digits))
    hl_cleansed.append(clean)
    
# View comparison
print('Original texts :')
print(df['headline'][1])
print('\nAfter cleansed :')
print(hl_cleansed[1])


# In[358]:


# Tokenization process
hl_tokens = []
for hl in hl_cleansed:
    hl_tokens.append(hl.split())

# View Comparison
index = 1
print('Before tokenization :')
print(hl_cleansed[index])
print('\nAfter tokenization :')
print(hl_tokens[index])


# In[359]:


#lemmatization
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Init Lemmatizer
lemmatizer = WordNetLemmatizer()

hl_lemmatized = []
for tokens in hl_tokens:
    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    hl_lemmatized.append(lemm)
    
# Example comparison
word_1 = ['skyrim','dragons', 'are', 'having', 'parties']
word_2 = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_1]
print('Before lemmatization :\t',word_1)
print('After lemmatization :\t',word_2)


# In[360]:


#preparing the data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Vectorize and convert text into sequences
max_features = 2000
max_token = len(max(hl_lemmatized))
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(hl_lemmatized)
sequences = tokenizer.texts_to_sequences(hl_lemmatized)
X = pad_sequences(sequences, maxlen=max_token)
print(X)


# In[361]:


#To check the convertion
index = 2
print('Before :')
print(hl_lemmatized[index],'\n')
print('After sequences convertion :')
print(sequences[index],'\n')
print('After padding :')
print(X[index])
X.shape


# In[362]:


#Spiliting of X to train the model
#As X was in concat form of both train and test file 
X1 = X[0:24209,:]
X2 = X[24209:,:]


# In[363]:


X1.shape


# In[364]:


#Spliting of the training and testing data to build a the model
from sklearn.model_selection import train_test_split

Y = df_train['is_sarcastic'].values
Y = np.vstack(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X1,Y,test_size=0.3, random_state = 42)


# In[365]:


Y.shape


# In[366]:


X2.shape


# In[367]:


X_train.shape


# In[368]:


Y_train.shape


# In[369]:


#model bulding
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

embed_dim = 64

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = max_token))
model.add(LSTM(96, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[370]:


#training process
epoch = 8
batch_size = 200
model.fit(X_train, Y_train, epochs = epoch, batch_size=batch_size, verbose = 2)


# In[371]:


#test model
loss, acc = model.evaluate(X_test, Y_test, verbose=2)
print("Overall scores")
print("Loss\t\t: ", round(loss, 3))
print("Accuracy\t: ", round(acc, 3))


# In[372]:


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_test)):
    
    result = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.around(result) == np.around(Y_test[x]):
        if np.around(Y_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.around(Y_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1


# In[373]:


print("Sarcasm accuracy\t: ", round(pos_correct/pos_cnt*100, 3),"%")
print("Non-sarcasm accuracy\t: ", round(neg_correct/neg_cnt*100, 3),"%")


# In[374]:


ypred = model.predict(X2)


# In[375]:


ypred1 = pd.DataFrame(ypred)


# In[376]:


ypred1.to_csv('pred.csv')

