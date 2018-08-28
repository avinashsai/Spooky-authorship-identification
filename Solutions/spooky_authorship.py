#Author : Avinash Madasu
import os
import re
import pandas as pd
import numpy as np
import gensim
import nltk

import zipfile

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopword = stopwords.words('english')


with zipfile.ZipFile('glove.42B.300d.zip', 'r') as myzip:
    myzip.extractall()

def load_glove_model():
  model = {}
  with open('glove.42B.300d.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
      splitline = line.split()
      word = splitline[0]
      embedding = np.array([float(val) for val in splitline[1:]])
      model[word] = embedding
    return model

glove_w2vmodel = load_glove_model()

train_data = pd.read_csv('train.csv')

train_data.head(3)

train_data.drop(["id"],axis=1,inplace=True)

def convert_to_labels():
  train_data["author"] = train_data["author"].mask(train_data["author"]=='EAP',0)
  train_data["author"] = train_data["author"].mask(train_data["author"]=='HPL',1)
  train_data["author"] = train_data["author"].mask(train_data["author"]=='MWS',2)
  train_labels = train_data['author'].astype(int)
  train_labels = train_labels.as_matrix()
  train_data.drop(['author'],axis=1,inplace=True)
  return train_labels

train_labels = convert_to_labels()

train_corpus = list(train_data["text"].as_matrix())
del train_data

max_sentence_length = 40

embedding_dim = 50

def preprocess(text): 
  text = re.sub(r"it\'s","it is",str(text))
  text = re.sub(r"i\'d","i would",str(text))
  text = re.sub(r"don\'t","do not",str(text))
  text = re.sub(r"he\'s","he is",str(text))
  text = re.sub(r"there\'s","there is",str(text))
  text = re.sub(r"that\'s","that is",str(text))
  text = re.sub(r"can\'t", "can not", text)
  text = re.sub(r"cannot", "can not ", text)
  text = re.sub(r"what\'s", "what is", text)
  text = re.sub(r"What\'s", "what is", text)
  text = re.sub(r"\'ve ", " have ", text)
  text = re.sub(r"n\'t", " not ", text)
  text = re.sub(r"i\'m", "i am ", text)
  text = re.sub(r"I\'m", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r"\'s"," is",text)
  text = re.sub(r"[0-9]"," ",str(text))
  sents = word_tokenize(text)
  return sents

train_length = len(train_corpus)

train_vectors = np.zeros((train_length,max_sentence_length,embedding_dim))

def pad_sentences(start,vectors):
  for i in range(start,max_sentence_length):
    vectors[i,:] = np.random.rand(50)

def convert_to_vectors(sentence):
  sents = preprocess(sentence)
  vectors = np.zeros((max_sentence_length,50))
  count = 0
  for sent in sents:
    if sent not in stopword:
      if sent in glove_w2vmodel:
        vector = glove_w2vmodel[sent]
        vectors[count,:] = vector[:50]
        count+=1
    if(count==max_sentence_length):
      return vectors
  if(count<max_sentence_length):
    pad_sentences(count,vectors)
    return vectors

def generate_train_vectors():
  i = 0
  for sentence in train_corpus:
    train_vectors[i,:,:] = convert_to_vectors(sentence)
    i =  i+1

generate_train_vectors()

import tensorflow as tf
import keras

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Input, Embedding, LSTM, Merge,Bidirectional
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU,Dense,Dropout,Lambda
from keras import metrics
from keras.layers.normalization import BatchNormalization

y_train  = keras.utils.to_categorical(train_labels,num_classes=3)

n_hidden1 = 64

def lstm_model():
  model = Sequential()
  model.add(LSTM(64,input_shape=(max_sentence_length,embedding_dim)))
  model.add(Dense(3,activation='softmax'))
  return model

model = lstm_model()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_vectors,y_train,batch_size=128,epochs=50)

test_data = pd.read_csv('test.csv')

test_corpus = list(test_data['text'].as_matrix())
test_data.drop(['text'],axis=1,inplace=True)

test_length = len(test_corpus)

test_vectors = np.zeros((test_length,max_sentence_length,embedding_dim))

def create_test_vectors():
  i = 0
  for sentence in test_corpus:
    test_vectors[i,:,:] = convert_to_vectors(sentence)
    i =  i+1

create_test_vectors()

pred = model.predict(test_vectors)

test_data['EAP'] = pred[:,0]

test_data['HPL'] = pred[:,1]
test_data['MWS'] = pred[:,2]

test_data.to_csv('submission.csv',header=True,index=False)


