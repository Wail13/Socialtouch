import pandas as pd
from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
 from itertools import  combinations,product
 from scipy.spatial import distance

Stimulus={'hug1_p':1,'hug2_p':2,'hug3_p':3,'str1_p':4,'str2_p':5,'hold1_p':6,'tap1_neu':7,
          'sha1_n':8,'sha2_n':9,'gri1_n':10,'nudg1_n':11,'nudg2_n':12,'slap1_n':13}


def bar(data):
    sizes=np.sum(data,axis=0)
    label=['hand','forearm','arm','shoul','chest','abs','uback','lback']
    
    plt.bar(label, sizes,label='Men')
    plt.savefig('pie.png')
    plt.show()
    
def do_classify(X,y,Xtest,ytest):
    importance=[]
    scores=[]
    for i in range(100):
        clf = ExtraTreesClassifier(n_estimators=10)
        clf = clf.fit(X, y)
        scores.append(clf.score(Xtest,ytest))
        importance.append(clf.feature_importances_)  
    
    mean_importance=np.mean(importance,axis=0)
    mean_scores=np.mean(scores)
    return mean_importance,mean_scores

df=pd.read_csv('/home/jan/Desktop/workspace/Social_touch/transform_annotation/data.csv',sep=',')
df=df.drop(columns=['Unnamed: 0'])
######################################################################################################
data=df
print(data.shape," #videos and #attributes")
print("########### touch action ######################")
print(np.unique(data.iloc[:,-5]))
######################################### Action histogram #############################################################
plt.hist(data.iloc[:,-5],bins=50)
print("Number of each action ")
##########################################Activated zones for men and women ###########################################################
n_groups = 8
means_men = np.sum(data.iloc[:,0:8],axis=0)
means_women = np.sum(data.iloc[:,8:16],axis=0)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, means_men, bar_width,
                alpha=opacity, color='b', error_kw=error_config,
                label='Men')

rects2 = ax.bar(index + bar_width, means_women, bar_width,
                alpha=opacity, color='r', error_kw=error_config,
                label='Women')

ax.set_xlabel('body parts')
ax.set_ylabel('#Number')
ax.set_title('Activated body parts by gender')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['hand','forearm','arm','shoul','chest','abs','l_leg','uback','lback'])
ax.legend()


fig.tight_layout()
plt.show()
#fig.savefig('activation.png')
####################################################Tree based selection #######################################################################


#keep=[4,5,6,7,13,14,15,16]
#data.iloc[:,-3:]=(data.iloc[:,-3:]-np.mean(data.iloc[:,-3:]))/np.std(data.iloc[:,-3:])
X_women_touched=data[data.INITIATIVE_F==0]
X_women_touched.drop(X_women_touched.columns[8:16], axis=1, inplace=True)
X_women_touched.columns.values[0:8]=['T0','T1','T2','T3','T4','T5','T6','T7']
X_women_touched['sex_touched']=0

X_men_touched=data[data.INITIATIVE_F==1]
X_men_touched.drop(X_men_touched.columns[0:8], axis=1, inplace=True)
X_men_touched.columns.values[0:8]=['T0','T1','T2','T3','T4','T5','T6','T7']
X_men_touched['sex_touched']=1
     
touched=pd.concat([X_women_touched,X_men_touched],axis=0,  ignore_index=True)   


X_touched=touched.iloc[:,0:8].values

########### action#############
y_action=touched.iloc[:,-6]
y_action=pd.get_dummies(y_action).values
y_action=np.argmax(y_action,axis=1)
################ Valence #####################
y_valence=touched.iloc[:,-4]
################### Arousal #####################
y_arousal=touched.iloc[:,-3]
###################  valence/arousal ###############################
y_va=touched.iloc[:,-4:-2].values
################# Motion energy ##################
y_ME=touched.iloc[:,-2]

####################################### Action-touched recognition from zones ##################################################################################
X_train, X_test, y_train, y_test = train_test_split(
    y_ME.reshape(-1,1), np.round(y_action), test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X_touched, np.round(y_action), test_size=0.33, random_state=42)

mean_importance_action,mean_scores_action=do_classify(X_train,y_train,X_test,y_test)
print(mean_importance_action)
print('#############  Accuracy of the classifier for touch_zones/actions(hug,slap,...)',mean_scores_action)
      
      

#####################################################################################################################
d=touched.groupby(['Stimulus']).mean()
          
d2=d[['T0','T1','T2','T3','T4','T5','T6','T7']]
d2=d2.values
d3=np.where(d2>=0.5,1,0)
plt.imshow(d2)
plt.set_xticklabels(['T0','T1','T2','T3','T4','T5','T6','T7'])




#####################################optimization ###################################################################

x=[0,1,1,0,1,1,0,1]

objectiv(d2,x)
def objectiv(d2,x):
    val=0
    for count,f in enumerate(x):
        if not f:
            d2[:,count]=0
        
    
    for d in combinations(list(range(0,8)), 2):
        i,j=d
        val=(distance.euclidean(d2[i,:],d2[j,:]))+val
        
    return val
 
    
all_val=[]   
x_all=[] 
for i in product([0,1], repeat=8):
    print(list(i))
    x_all.append(list(i))
    if(np.sum(list(i)) >= 3):
        all_val.append(objectiv(d2.copy(),list(i))/(np.sum(list(i))+1))



