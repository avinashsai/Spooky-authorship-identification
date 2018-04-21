import os
import re
import pandas as pd
import numpy as np
import sklearn
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopword = stopwords.words('english')

!pip install gensim

train_data = pd.read_csv('train.csv',sep=',')

print(train_data.head(10))

train_data = train_data.drop(["id"],axis=1)

train_length = len(train_data)
print("Train length is : ",train_length)

test_data = pd.read_csv('test.csv',sep=',')

print(test_data.head(3))

test_length = len(test_data)
print("Test length is : ",test_length)

train_labels = train_data["author"].as_matrix()

train_corpus = train_data["text"].as_matrix()

test_corpus = test_data["text"].as_matrix()

corpus = []

def preprocess(sentence):
  sentence = re.sub(r'[^\w\s]+'," ",str(sentence))
  sentence = re.sub(r'[^a-zA-Z]+'," ",str(sentence))
  sents = word_tokenize(sentence)
  new_sents = " "
  for i in range(len(sents)):
    if(sents[i].lower() not in stopword):
      new_sents+=sents[i].lower()+" "
  return new_sents

corpus_train = []

for i in range(train_length):
  sent = preprocess(train_corpus[i])
  corpus_train.append(sent)
  corpus.append(sent.split())

corpus_test = []

for i in range(test_length):
  sent = preprocess(test_corpus[i])
  corpus_test.append(sent)
  corpus.append(sent.split())

print(corpus_train[0:2])

print(corpus_test[0:2])

corpus[0:2]

model_word2vec = gensim.models.Word2Vec(sg=1,seed=1,size=100, min_count=1, window=10,sample=1e-4)

model_word2vec.build_vocab(corpus)

model_word2vec.train(corpus,total_examples=model_word2vec.corpus_count,epochs=50)

model_word2vec.save('spooky_trained_vecs.txt')

model_word2vec = Word2Vec.load('spooky_trained_vecs.txt')

vocabulary = model.wv.vocab

train_vectors = np.zeros((train_length,100))

train = []

total = 0

for i in range(train_length):
  vec = np.zeros(100)
  count = 0
  sents = corpus_train[i].split()
  for j in range(len(sents)):
    if(sents[j] in model_word2vec.wv.vocab):
      vec = vec + model_word2vec.wv[sents[j]]
      count+=1
      
  if(count==0):
    total+=1
    train_vectors[i] = np.random.random_sample([100])
  else:
    train_vectors[i] = vec/count
  train.append([vec,train_labels[i]])

test_vectors =  np.zeros((test_length,100))

for i in range(test_length):
  vec = np.zeros(100)
  count = 0
  sents = corpus_test[i].split()
  for j in range(len(sents)):
    if(sents[j] in model_word2vec.wv.vocab):
      vec = vec + model_word2vec.wv[sents[j]]
      count+=1
      
  if(count==0):
    total+=1
    vec = np.random.random_sample([100])
  else:
    vec = vec/count
  test_vectors[i] = vec

train_labs = np.zeros(train_length)

for i in range(train_length):
  if(train_labels[i]=="EAP"):
    train_labs[i]=0
  elif(train_labels[i]=="HPL"):
    train_labs[i]=1
  else:
    train_labs[i]=2

train_vecs = train_vectors.reshape((train_length,100,1))

test_vecs = test_vectors.reshape((test_length,100,1))

import keras
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras import metrics
from keras.layers import advanced_activations
from keras import optimizers

train_lab = keras.utils.to_categorical(train_labs,num_classes=3)

train_lab[0]

model = Sequential()
model.add(Conv1D(64,kernel_size=3,input_shape=(100,1)))
model.add(Conv1D(32,kernel_size=3))
model.add(Dropout(0.25))
model.add(Conv1D(16,kernel_size=3))
model.add(Conv1D(8,kernel_size=3))
model.add(Conv1D(4,kernel_size=3))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50,activation='tanh'))
model.add(Dense(30,activation='tanh'))
model.add(Dense(15,activation='tanh'))
model.add(Dense(8,activation='tanh'))
model.add(Dense(3,activation='softmax'))

model.summary()

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(train_vecs,train_lab,batch_size=64,epochs=75)

test_vals = model.predict(test_vecs)

import csv

with open('submission.csv', 'w') as csvfile:
    fieldnames = ['id', 'EAP','HPL','MWS']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for predict in range(test_length):
      writer.writerow({'id': test_data["id"][predict], 'EAP': test_vals[predict][0],'HPL':test_vals[predict][1],'MWS':test_vals[predict][2]})

files.download('submission.csv')





