import os
import re
import pandas as pd
import numpy as np
import gensim
import nltk
import zipfile

import tensorflow as tf
import keras

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Input, Embedding, LSTM, Merge,Bidirectional
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU,Dense,Dropout,Lambda,Flatten
from keras import metrics
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D,MaxPooling2D,Conv1D,MaxPooling1D



nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopword = stopwords.words('english')

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

max_sentence_length = 100

embedding_dim = 100

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
  text = re.sub(r"[^a-zA-Z]"," ",str(text))
  sents = word_tokenize(text)
  new_sentence = " "
  for word in sents:
    if word.lower() not in stopword:
      new_sentence+=word.lower()+" "
  return new_sentence

train_length = len(train_corpus)

total_corpus = []

train_sent = []

for sentence in train_corpus:
  sentence = preprocess(sentence)
  train_sent.append(sentence)
  total_corpus.append(sentence.split())

test_data = pd.read_csv('test.csv')

test_length = len(test_data)

print(test_length)

test_corpus = list(test_data['text'].as_matrix())
test_data.drop(['text'],axis=1,inplace=True)

test_sent = []

for sentence in test_corpus:
  sentence = preprocess(sentence)
  test_sent.append(sentence)
  total_corpus.append(sentence.split())

model_fasttext = gensim.models.FastText(sg=1,seed=1,size=100, min_count=1, window=10,sample=1e-4)

model_fasttext.build_vocab(total_corpus)

model_fasttext.train(total_corpus,total_examples=model_word2vec.corpus_count,epochs=50)

vocabulary = model_fasttext.wv.vocab

train_vectors = np.zeros((train_length,max_sentence_length,embedding_dim))

def convert_to_vectors(sentence):
  vectors = np.zeros((max_sentence_length,embedding_dim))
  count = 0
  words = sentence.split()
  for word in words:
    if word in vocabulary:
      vectors[count,:] = model_fasttext[word]
      count+=1
    if(count>=max_sentence_length):
      break
  return vectors

def generate_train_vectors():
  i = 0
  for sentence in train_corpus:
    train_vectors[i,:,:] = convert_to_vectors(sentence)
    i =  i+1

generate_train_vectors()

test_vectors = np.zeros((test_length,max_sentence_length,embedding_dim))

def generate_test_vectors():
  i = 0
  for sentence in test_corpus:
    test_vectors[i,:,:] = convert_to_vectors(sentence)
    i =  i+1

generate_test_vectors()

y_train  = keras.utils.to_categorical(train_labels,num_classes=3)

n_hidden1 = 16

n_hidden2 = 4

def dense_model():
  model = Sequential()
  model.add(Conv1D(150,3,activation='relu',input_shape=(max_sentence_length,embedding_dim)))
  model.add(MaxPooling1D(2))
  model.add(Dropout(0.5))
  model.add(Conv1D(100,3,activation='relu'))
  model.add(MaxPooling1D(2))
  model.add(Dropout(0.5))
  model.add(Conv1D(70,3,activation='relu'))
  model.add(MaxPooling1D(2))
  model.add(Dropout(0.5))
  model.add(Conv1D(30,3,activation='relu'))
  model.add(MaxPooling1D(2))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(70,activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(30,activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(8,activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(3,activation='softmax'))
  return model

model = dense_model()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_vectors,y_train,batch_size=32,epochs=50)

pred = model.predict(test_vectors)

test_data['EAP'] = pred[:,0]

test_data['HPL'] = pred[:,1]
test_data['MWS'] = pred[:,2]

test_data.to_csv('fasttext_1dcnn.csv',header=True,index=False)


