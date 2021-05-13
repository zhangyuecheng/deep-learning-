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

vocabulary_size = 400000
#***********
time_step=300
embedding_size=100

dataVal_Fake_Real=pd.read_csv('/home/zhangyc/下载/paper/data/fake_or_real_news.csv')

texts=[]
texts=dataVal_Fake_Real['text']#####################################
label=dataVal_Fake_Real['label']
#print(label)
#X=texts.astype(str).values.tolist()
#X=np.reshape(X,(-1,1))
from Clean_Texts import clean_text
X=texts.map(lambda x: clean_text(x))
#print(X)
#label=label.astype(int).values
labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

training_size=int(0.8*X.shape[0])
print(X.shape[0],training_size)
X_train=X[:training_size]
y_train=y[:training_size]
X_test=X[training_size:]
y_test=y[training_size:]

#Tokenizing texts
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)
sequences_train= tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(sequences_train, maxlen=time_step,padding='post')

print(len(tokenizer.word_index))


vocab_size=len(tokenizer.word_index)+1

#Reading Glove
f = open('/home/zhangyc/下载/paper/code/C-LSTM/glove.6B.100d.txt',encoding='utf-8')
embeddings={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_size))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape[0],embedding_matrix.shape[1])


sequences_test= tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(sequences_test, maxlen=time_step,padding='post')

#print(len(X_test),len(y_test))
#print(label)


# Embedding
#maxlen = 100
#embedding_size = 32

start =time.time()

## create model
model = Sequential()
model.add(Embedding(np.array(embedding_matrix).shape[0],
                          embedding_size, weights=[embedding_matrix], trainable=False))

model.add(LSTM(300))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
## Fit train data
history=model.fit(X_train, y_train, epochs = 5,batch_size=64,shuffle=True)


#print("Saving Model...")
#model_name = 'Models/Model_lstm_FR_2.h5'########################################3
#model.save(model_name)#################################################

score=model.evaluate(X_test,y_test,verbose=1)
print('acc: '+str(score[1]))

from sklearn.metrics import precision_recall_fscore_support,classification_report
y_pred=model.predict_classes(X_test)
end =time.time()
print('LSTM Classification report:\n',classification_report(y_test,y_pred))
#print('Classification report:\n',precision_recall_fscore_support(y_test,y_pred))
#print(y_pred)

'''
model = load_model('Models/Model_lstm_FR_2.h5')
model.name='Model_lstm_FR_2.h5'
'''

fpr,tpr,threshold = roc_curve(y_test, y_pred) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LSTM')
plt.legend(loc="lower right")
plt.show()


print('Running time: %s Seconds'%(end-start))

