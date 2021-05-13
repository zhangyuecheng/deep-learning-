import os
from sklearn.preprocessing import LabelEncoder
from keras.regularizers import l2
from Clean_Texts import clean_text

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM,Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import optimizers
from tensorflow.keras.layers import TimeDistributed
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import time



import nltk
nltk.download('punkt')


dataset = pd.read_csv('/home/zhangyc/下载/paper/data/All.csv')
print(dataset.shape)

texts=[]
texts=dataset['Statement']#####################################
label=dataset['Label']

labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

training_size=int(0.8*dataset.shape[0])
print(dataset.shape[0],training_size)
data_train=dataset[:training_size]['Statement']
y_train=y[:training_size]
data_rest=dataset[training_size:]['Statement']
y_test=y[training_size:]


MAX_SENT_LENGTH = 100
MAX_SENTS = 20
MAX_NB_WORDS = 400000
EMBEDDING_DIM = 100
#VALIDATION_SPLIT = 0.2

vocabulary_size = 400000
time_step=300
embedding_size=100
# Convolution
filter_length = 3
nb_filters = 128
n_gram=3
cnn_dropout=0.0
nb_rnnoutdim=300
rnn_dropout=0.0
nb_labels=1
dense_wl2reg=0.0
dense_bl2reg=0.0


texts=data_train

texts=texts.map(lambda x: clean_text(x))

tokenizer=Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(texts)
encoded_train=tokenizer.texts_to_sequences(texts=texts)
vocab_size_train = len(tokenizer.word_index) + 1
print(vocab_size_train)

x_train = sequence.pad_sequences(encoded_train, maxlen=time_step,padding='post')



texts=data_rest

texts=texts.map(lambda x: clean_text(x))


encoded_test=tokenizer.texts_to_sequences(texts=texts)

x_test = sequence.pad_sequences(encoded_test, maxlen=time_step,padding='post')



GLOVE_DIR = "."
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding='utf-8')
embeddings_train={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_train[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_train))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size_train, embedding_size))
for word, i in tokenizer.word_index.items():
	embedding_vector_train = embeddings_train.get(word)
	if embedding_vector_train is not None:
		embedding_matrix[i] = embedding_vector_train

start =time.time()

model = Sequential()
model.add(Embedding(vocab_size_train, embedding_size, input_length=time_step,
                    weights=[embedding_matrix],trainable=False))
model.add(Conv1D(filters=nb_filters,
                 kernel_size=n_gram,
                 padding='valid',
                 activation='relu'))
if cnn_dropout > 0.0:
    model.add(Dropout(cnn_dropout))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(nb_rnnoutdim))
if rnn_dropout > 0.0:
    model.add(Dropout(rnn_dropout))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train,
          epochs=5, batch_size=64)


score=model.evaluate(x_test,y_test,verbose=1)
print('acc: '+str(score[1]))

from sklearn.metrics import precision_recall_fscore_support,classification_report
y_pred=model.predict_classes(x_test)
end=time.time()

print('C_LSTM Classification report:\n',classification_report(y_test,y_pred))
#print('Classification report:\n',precision_recall_fscore_support(y_test,y_pred))
#print(y_pred)

fpr,tpr,threshold = roc_curve(y_test, y_pred) 
roc_auc = auc(fpr,tpr) 

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('C_LSTM')
plt.legend(loc="lower right")
plt.show()


print('Running time: %s Seconds'%(end-start))










































































