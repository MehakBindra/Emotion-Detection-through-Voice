#!/usr/bin/env python
# coding: utf-8
import IPython.display as ipd
import os
import numpy as np
import pandas as pd
import librosa
import glob
import librosa.display
import matplotlib.pyplot as plt
directory='E:/bdann audio/'

from glob import glob
glob(directory)

#Actor_01
folder_name='Actor_0'
all_=[]
for i in range(1,25):
    for root, dirs, files in os.walk(directory+folder_name+str(i)+'/'):
        all_.append(files)

#Actor_01
folder_name='Actor_1'
for i in range(0,10):
    for root, dirs, files in os.walk(directory+folder_name+str(i)+'/'):
        all_.append(files)

#Actor_01
folder_name='Actor_2'
for i in range(0,5):
    for root, dirs, files in os.walk(directory+folder_name+str(i)+'/'):
        all_.append(files)

file_names=np.array(all_)

file_names=file_names.reshape(-1,1)

df=pd.DataFrame(file_names,columns=['File_name'])

df.head() #Modality*-speech/song*-emotion-emotional intensity-statement-repetition*-actor

df['emotion']=""
df['emotional intensity']=""
df["statement"]=""
df["actor"]=""
df['sex']=""

for i in range(len(df)):
    df.loc[i,'emotion']=df.loc[i,"File_name"][6:8]
    df.loc[i,'emotional intensity']=df.loc[i,"File_name"][9:11]
    df.loc[i,'statement']=df.loc[i,"File_name"][12:14]
    df.loc[i,'actor']=df.loc[i,"File_name"][-6:-4]
    if(int(df.loc[i,'actor'])%2==0):
        df.loc[i,'sex']="Female"
    else:
        df.loc[i,'sex']="Male"


all_list=[]
all_dict={}
folder_name='Actor_0'
all_=[]
for i in range(1,10):
    for root, dirs, files in os.walk(directory+folder_name+str(i)+'/'):
        for name in files:
            all_list.append(directory+folder_name+str(i)+'/'+name)

#Actor_01
folder_name='Actor_1'
for i in range(0,10):
    for root, dirs, files in os.walk(directory+folder_name+str(i)+'/'):
         for name in files:
            all_list.append(directory+folder_name+str(i)+'/'+name)

#Actor_01
folder_name='Actor_2'
for i in range(0,5):
    for root, dirs, files in os.walk(directory+folder_name+str(i)+'/'):
        for name in files:
            all_list.append(directory+folder_name+str(i)+'/'+name)

df_features=pd.DataFrame()
df_temp=pd.DataFrame()

for path in all_list:
    y, sr = librosa.load(path)
    sr=np.array(sr)
    mfccs=np.mean(librosa.feature.mfcc(y=y,n_mfcc=25),axis=0)
    df_temp=pd.DataFrame([-mfccs/100])
    df_temp=df_temp.iloc[:,0:130]
    df_features=df_features.append(df_temp,sort=False)

df_features=df_features.reset_index()
df_features.drop('index',axis=1,inplace=True)
df_features_cols=list('mf'+str(x) for x in range(0,130))
df_features.columns=df_features_cols

df_features['MFCC_Mean']=df_features[df_features_cols].mean(axis=1)
df_features['MFCC_Var']=df_features[df_features_cols].var(axis=1)
df_features['Max']=df_features[df_features_cols].max(axis=1)
df_features['Min']=df_features[df_features_cols].min(axis=1)
df_features['Range']=df_features['Max']-df_features['Min']

df_features.dropna()
df_features_med=pd.DataFrame()

for path in all_list:
    y, sr = librosa.load(path)
    S = np.abs(librosa.stft(y))
    med_pow=librosa.power_to_db(S**2, ref=np.median)
    med_pow=np.mean(med_pow,axis=0)
    df_temp=pd.DataFrame([med_pow])
    df_temp=df_temp.iloc[:,0:130]
    df_features_med=df_features_med.append(df_temp,sort=False)

df_features_med=df_features_med.reset_index()
df_features_med.drop('index',axis=1,inplace=True)
df_features_med_cols=list('med'+str(x) for x in range(0,130))
df_features_med.columns=df_features_med_cols
df_features_med['Med_Mean']=df_features_med[df_features_med_cols].mean(axis=1)
df_features_med['Med_Var']=df_features_med[df_features_med_cols].var(axis=1)
df_features_med['Med_Max']=df_features_med[df_features_med_cols].max(axis=1)
df_features_med['Med_Min']=df_features_med[df_features_med_cols].min(axis=1)
df_features_med['Med_Range']=df_features_med['Med_Max']-df_features_med['Med_Min']


df_features_peak=pd.DataFrame()
for path in all_list:
    y, sr = librosa.load(path)
    sr=np.array(sr)
    S = np.abs(librosa.stft(y))

    peak_pow=librosa.power_to_db(S**2, ref=np.max)
    peak_pow=np.mean(peak_pow,axis=0)
    df_temp=pd.DataFrame([peak_pow])
    df_temp=df_temp.iloc[:,0:130]
    df_features_peak=df_features_peak.append(df_temp,sort=False)

df_features_peak=df_features_peak.reset_index()
df_features_peak.drop('index',axis=1,inplace=True)
df_features_peak=df_features_peak*-1
df_features_peak_cols=list('peak'+str(x) for x in range(0,130))
df_features_peak.columns=df_features_peak_cols
df_features_peak['Med_Mean']=df_features_peak[df_features_peak_cols].mean(axis=1)
df_features_peak['Med_Var']=df_features_peak[df_features_peak_cols].var(axis=1)
df_features_peak['Med_Max']=df_features_peak[df_features_peak_cols].max(axis=1)
df_features_peak['Med_Min']=df_features_peak[df_features_peak_cols].min(axis=1)
df_features_peak['Med_Range']=df_features_peak['Med_Max']-df_features_peak['Med_Min']

df_features_all=pd.concat([df_features,df_features_med,df_features_peak],axis=1, sort=False)

df_features.to_csv('Mfccs_Features.csv')
df=pd.concat([df,df_features_all],axis=1, sort=False)
df.to_csv('FeaturesAll.csv',index=False)
