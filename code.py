#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.utils import np_utils

df=pd.read_csv('FeaturesAll.csv')

df=df.dropna()

X=df.drop('File_name',axis=1).drop('emotion',axis=1).drop('statement',axis=1).drop('actor',axis=1).drop('Min',axis=1).drop('Max',axis=1).drop('Med_Min',axis=1).drop('Med_Max',axis=1)

y=df['emotion']

temp=pd.get_dummies(df['sex'])
X['gender']=temp['Male']
X=X.drop('sex',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
y_train=y_train.values
X_train=X_train.values
X_test=X_test.values

from sklearn.metrics import confusion_matrix
from keras.models import Sequential

y_train = pd.get_dummies(y_train)

import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation

model=Sequential()

model.add(Conv1D(128,5,padding='same',input_shape=(403,1)))
model.add(Activation('relu'))
model.add(Conv1D(128,5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128,5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128,5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128,5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Conv1D(128,5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(8,activation='softmax'))
opt=keras.optimizers.rmsprop(lr=0.00005,decay=1e-6)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train.reshape(-1,403,1),y_train,epochs=500,batch_size=5)

y_test=pd.get_dummies(y_test)


model.evaluate(x=X_test.reshape(-1,403,1),y=y_test)
model.save("model.h5")
