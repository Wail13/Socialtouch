#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:49:54 2018

@author: wail
"""
from keras.utils.np_utils import to_categorical
import pandas as pd 
import numpy as np 
#############################################################################################
df=pd.read_csv('/home/jan/Desktop/workspace/Social_touch/transform_annotation/rep2.csv')
df_info=pd.read_csv('/home/jan/Desktop/workspace/Social_touch/transform_annotation/Annot.csv')

#############################valence arousal naturalness ########################################
x1=pd.ExcelFile('/home/jan/Desktop/workspace/Social_touch/transform_annotation/Experiment2_DATA.xlsx')
x2=pd.ExcelFile('/home/jan/Desktop/workspace/Social_touch/transform_annotation/MotionEnergy_DATA.xlsx')

df_VAN = (x1.parse(x1.sheet_names)) 
m=list(df_VAN.values())
Van=pd.concat(m, axis=0)
df_ME = (x2.parse('Sheet1')) 
gvan=Van.groupby(['Actors-Pair','Stimulus']).mean().reset_index()
gME=df_ME.groupby(['Actors-Pair','Stimulus']).mean().reset_index()
################################################################################################  


################################################################################################  

body_zones={'Hand':0,'Forearm':1,'Arm':2,'Shoulder':3,'Chest':4,'Abs':5,'upperback':6,'lowerback':7}
videos_selected=[2,3,4,5,7,8,9,13,15,16,17,18,20,21,22,26,28,29,30,31,33,34,35]
Stimulus={'hug1_p':1,'hug2_p':2,'hug3_p':3,'str1_p':4,'str2_p':5,'hold1_p':6,'tap1_neu':7,
          'sha1_n':8,'sha2_n':9,'gri1_n':10,'nudg1_n':11,'nudg2_n':12,'slap1_n':13}

######################################################################################################
for d in body_zones:
    body_zones[d]= to_categorical(body_zones[d],num_classes=8)

def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
    
    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    
    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return zip(*[lst[i::n] for i in range(n)]) 

def annotator_data(data):
    ########## interaction data for single annotator####################################
    interactions=[ (data[data.columns[i]].tolist(),data[data.columns[i+1]].tolist())  for i,j in group(range(6,53),2)]

    annotator_inter_list=[]
    ############## single interaction ##########################
    for interaction in interactions:
        Male,Female= interaction        
        #### Activations Names######
        Male_activations=[d.replace(" ", "") for d in Male[0].split(",")]
        Female_activations=[d.replace(" ", "") for d in Female[0].split(",")]

        ###### Activation onehotencoding #############
        Male_oneHot=np.sum([body_zones[d] for d in Male_activations],axis=0)
        Female_oneHot=np.sum([body_zones[d] for d in Female_activations],axis=0)
        #######################################################
        interaction_vect=np.concatenate((Female_oneHot, Male_oneHot), axis=0)
        annotator_inter_list.append(interaction_vect)
    return annotator_inter_list




def all_data(df):
    dataframe_list=[]
    for i in range(1,df.shape[0]) :
        data=df[i:i+1]
        ddf=pd.DataFrame(annotator_data(data))
        ddf['name']=data[data.columns[1]].tolist()[0]
        ####### concat videos infos ##########
        df_videos=df_info[df_info['VIDEO_NUM'].isin(videos_selected)]
        videos_infos=df_videos[[df_videos.columns[i] for i in [0,1,2,3,22,23,24,25,25]]]
        ###### ignore index
        ddf.reset_index(drop=True, inplace=True)
        videos_infos.reset_index(drop=True, inplace=True)
        ########### concat infos and annotations
        ddf2=pd.concat([ddf,videos_infos],axis=1,  ignore_index=True)   
        
        dataframe_list.append(ddf2)
    
    all_data=pd.concat(dataframe_list,axis=0)
    return all_data





all_data=all_data(df)
all_data = all_data.iloc[:, :-2]
all_data.iloc[:, -1]=[Stimulus[S.lower()] for S in list(all_data.iloc[:, -1].values)]
all_data.columns=['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'M0', 'M1', 'M2', 'M3',
       'M4', 'M5', 'M6', 'M7', 'NAME', 'VIDEO_NUM', 'Actors-Pair', 'INITIATIVE_F',
       'INITIATIVE_M', 'ME', 'ACTION', 'Stimulus']

merged = pd.merge(all_data,gvan, on=['Stimulus','Actors-Pair'])
merged = pd.merge(merged,gME, on=['Stimulus','Actors-Pair'])
all_data=merged.drop(['ME','RT','RT.1'],axis=1)
all_data.to_csv('/home/jan/Desktop/workspace/Social_touch/transform_annotation/data.csv', sep=',')
