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

dataVal_Fake_Real=pd.read_csv('/home/zhangyc/下载/paper/data/All.csv')

texts=[]
texts=dataVal_Fake_Real['Statement']#####################################
label=dataVal_Fake_Real['Label']

from Clean_Texts import clean_text
X=texts.map(lambda x: clean_text(x))

labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))


training_size=int(0.8*X.shape[0])
X_train=X[:training_size]
y_train=y[:training_size]
X_test=X[training_size:]
y_test=y[training_size:]

vocabulary_size = 400000

time_step=300
embedding_size=100

tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)
sequences_train= tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(sequences_train, maxlen=time_step,padding='post')

print(len(tokenizer.word_index))


vocab_size=len(tokenizer.word_index)+1

f = open('glove.6B.100d.txt',encoding='utf-8')
embeddings={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings))

embedding_matrix = np.zeros((vocab_size, embedding_size))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape[0],embedding_matrix.shape[1])


sequences_test= tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(sequences_test, maxlen=time_step,padding='post')

filter_length = 3
nb_filter = 128

start =time.time()

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=time_step,
                    weights=[embedding_matrix],trainable=False))

model.add(Conv1D(filters=nb_filter,
                        kernel_size=filter_length,
                        activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=64,epochs=5)


print("Saving Model...")

score=model.evaluate(X_test,y_test,verbose=1)
print('acc: '+str(score[1]))

from sklearn.metrics import precision_recall_fscore_support,classification_report
y_pred=model.predict_classes(X_test)
end=time.time()
print('CNN Classification report:\n',classification_report(y_test,y_pred))

'''
model = load_model('/home/zhangyc/下载/paper/code/CNN/Models/Model_cnn_FR_2.h5')
model.name='Model_cnn_FR_2.h5'
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
plt.title('CNN')
plt.legend(loc="lower right")
plt.show()


print('Running time: %s Seconds'%(end-start))


