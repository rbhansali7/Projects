
# coding: utf-8

# ### Code for reading the Equivalence Classes and storing them in a dictionary.

# In[430]:

import sys
import re
import os
import pandas as pd
import csv
import numpy as np
import matplotlib as plt
from scipy import linalg 
import pickle
get_ipython().magic('matplotlib inline')

eqClasses_dict = {}

def storeInDict(line, person):
    
    s= line.split()
    n=len(s)
    eqClassLength = int(s[0])
    tup = list()
    
    #Storing length 1 equivalence class as a string instead of a tuple in the dictionary
    if(eqClassLength == 1):
        t_name = s[1]
        if t_name not in eqClasses_dict:
            eqClasses_dict[t_name] = [0 for i in range(369)]
            eqClasses_dict[t_name][person] = int(s[n-1])
        else:
            eqClasses_dict[t_name][person] = int(s[n-1])

    else:
        #Create the tuple of transcript IDs which will be the key
        for i in range(1,eqClassLength+1):
            tup.append(s[i])
        tup = tuple(tup)

        if tup not in eqClasses_dict:
            eqClasses_dict[tup] = [0 for i in range(369)]
            eqClasses_dict[tup][person] = int(s[n-1])
        else:
            eqClasses_dict[tup][person] = int(s[n-1])


# In[2]:

def parseInput(path, personCount):

    file = open(path,"r")

    lineCount = 0
    for line in file:
        if lineCount == 0:
            num_transcripts = int(line)
            lineCount+=1
            continue
        if lineCount == 1:
            num_eqClasses = int(line)
            lineCount+=1
            continue
        if lineCount < 2+ num_transcripts:
            lineCount+=1
            continue
        else:
            storeInDict(line, personCount)
            lineCount+=1
            continue


# In[4]:

def parseAllFiles():
    
    traincsv = pd.read_csv("/Users/jatingarg/Desktop/CompBioData/project1/p1_train.csv",low_memory=False)
    accessionList = traincsv['accession'].values
    
    for i in range(len(accessionList)):
        name = accessionList[i]
        s="/Users/jatingarg/Desktop/CompBioData/project1/train/"+str(name)+"/bias/aux_info/eq_classes.txt"
        parseInput(s, i)

    print (len(eqClasses_dict))
    count =0
#     for key in eqClasses_dict.keys():
#         print (key)
#         print (eqClasses_dict[key])
#         count+=1
#         if count>20:
#             break


# ### Resultant vector showing the numReads in each equivalence class that has existed in any of the individuals

# In[5]:

parseAllFiles()


# In[6]:

df = pd.DataFrame.from_dict(eqClasses_dict)


# ### The dataframe showing the equivalence class name as columns and numReads in each equivalence class for every individual

# In[7]:

df.head(10)


# In[146]:

df.to_csv("eq.csv",index=False)


# In[8]:

colList = df.columns


# In[9]:

df.shape


# In[10]:

readList = list()
for i in range(df.shape[1]):
    num = (df[colList[i]] != 0).sum()
    tup = tuple([num,i])
    readList.append(tup)


# In[11]:

sortedList = list()
revSortedList = list()


# In[12]:

sortedList = sorted(readList, key=lambda x:x[0])


# In[13]:

revSortedList = sorted(readList, key=lambda x:x[0], reverse=True)


# In[14]:

sortedList[10]


# ### Here we have sorted the the class according to whether the class is present in how many of the individuals

# In[15]:

sortedNp = np.array(sortedList)


# In[16]:

reverseSortedNp = np.array(revSortedList)


# In[17]:

sortedNp.mean()


# In[18]:

sortedDF = pd.DataFrame(sortedNp)


# In[19]:

sortedDF.head(2)


# ### Below result shows that there are some common Equivalence classes that are present in all 369 individuals and some are uniquely mapped to 1 person

# In[20]:

sortedDF[1].describe()


# In[22]:

tempDF = sortedDF.copy()


# In[24]:

tempDF = SVD(tempDF)


# In[23]:

def SVD(tempDF):
    featureMatrix=np.array(tempDF)    

    U, s, Vh = linalg.svd( featureMatrix, full_matrices=1, compute_uv=1 )
    low_dim_p = 10000
    return U[:,0:low_dim_p]


# In[412]:

sortedDF = tempDF[tempDF[0] >= 350]
sortedDF = sortedDF[sortedDF[0] <= 369]


# In[413]:

sortedDF.shape


# In[414]:

uniqueDF = sortedDF


# In[415]:

uniqueDF.head(2)


# In[416]:

uniqueDF.shape


# In[417]:

colIndex = uniqueDF[1].tolist()


# In[418]:

colIndex[0]


# In[419]:

colNames = list()


# In[420]:

for i in range(len(colIndex)):
    colNames.append(colList[colIndex[i]])


# In[421]:

dfTest1 = df.filter(colNames)


# In[422]:

dfTest1 = dfTest1.head(369)
dfTest1.shape


# ### Here we have made a visual representation showing number of equivalence classes present on y-axis and the number of persons having those number of equivalence classes common to them.

# #### So this shows more than .2 million equivalence classes are uniquely mapped to single persons while around 90K are common in all persons

# In[246]:

sortedDF[0].hist(figsize=(10,10),bins=500)


# ## Population Label Model

# In[423]:

traincsv = pd.read_csv("/Users/jatingarg/Desktop/CompBioData/project1/p1_train_pop_lab.csv",low_memory=False)
accessionList = traincsv['accession'].values
countryList = traincsv['population'].values


# In[424]:

traincsv.columns


# In[425]:

len(countryList)


# In[426]:

countryList[200]


# In[427]:

dfTest1["Country"] = countryList


# ### So here we choose only those equivalence classes that are common in most of the persons to train our model that is Random Forest and we applied 5 fold Cross Validation

# In[38]:

from sklearn.ensemble import RandomForestClassifier


# In[428]:

cross_val = 5
averageAccuracyScore = 0
f1pop = 0
rf_modelPop = RandomForestClassifier(n_estimators=100, # Number of trees
                                    max_features=dfTest1.shape[1]-10,    # Num features considered
                                    oob_score=False)    # Use OOB scoring*
for i in range(cross_val):
    msk = np.random.rand(len(dfTest1)) < 0.80

    train = dfTest1[msk]
    
    test = dfTest1[~msk]
    
    # Train the model
    rf_modelPop.fit(train.filter(dfTest1.columns[:-1]), train.filter(dfTest1.columns[-1:]))

    predictedDF = rf_modelPop.predict(test.filter(dfTest1.columns[:-1]))
    ansList = test['Country'].tolist()
    averageAccuracyScore += printAccuracy(ansList,predictedDF)
    f1pop += fScore(ansList,predictedDF)
    print(averageAccuracyScore)
print("F1-Score For Population label is : ",f1pop/float(cross_val))
print("Average Accuracy of model is " + str(averageAccuracyScore/cross_val))


# ### The results shows the average accuracy for the test data set i.e around 84.5 %

# In[377]:

predictedDF


# In[56]:

from sklearn.metrics import confusion_matrix
import sklearn.metrics


# ### This shows the confusion matrix for the classes i.e number of classes(countries) predicted correctly/incorrectly
# ####                    [[ 7,  1,  0,  0,  2],
# ####                    [ 0, 12,  1,  4,  1],
# ####                    [ 0,  0,  9,  0,  0],
# ####                    [ 7,  0,  0,  9,  0],
# ####                    [ 0,  0,  0,  0, 12]])
# #### Here we can see this model is not performing well specifically for the "TSI"

# In[379]:

confusion_matrix(ansList,predictedDF,labels=['GBR','FIN','CEU','TSI','YRI'])


# In[403]:

def printAccuracy(ansList,predictedDF):
    count = 0
    for i in range(len(ansList)):
        if ansList[i] == predictedDF[i]:
            count += 1
    return (count/len(ansList))


# ## Sequencing Center Model

# In[405]:

traincsv = pd.read_csv("/Users/jatingarg/Desktop/CompBioData/project1/p1_train_pop_lab.csv",low_memory=False)
accessionList = traincsv['accession'].values
sequencingCenterList = traincsv['sequencing_center'].values


# In[406]:

traincsv.columns


# In[407]:

dfTest1["SequencingCenter"] = sequencingCenterList


# In[408]:

def fScore(y_true,y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred,average='micro')


# In[411]:

cross_val = 5
averageAccuracyScore = 0
f1Seq = 0
rf_modelSeq = RandomForestClassifier(n_estimators=100, # Number of trees
                                    max_features=dfTest1.shape[1]-10,    # Num features considered
                                    oob_score=False)    # Use OOB scoring*
for i in range(cross_val):
    msk = np.random.rand(len(dfTest1)) < 0.80

    train = dfTest1[msk]
    
    test = dfTest1[~msk]
    
    # Train the model
    rf_modelSeq.fit(train.filter(dfTest1.columns[:-1]), train.filter(dfTest1.columns[-1:]))

    predictedDF = rf_modelSeq.predict(test.filter(dfTest1.columns[:-1]))
    ansList = test['SequencingCenter'].tolist()
    averageAccuracyScore += printAccuracy(ansList,predictedDF)
    f1Seq += fScore(ansList,predictedDF)
    print(averageAccuracyScore)
print("F1-Score For Sequencing Center: ",f1Seq/float(cross_val))
print("Average Accuracy of model is " + str(averageAccuracyScore/cross_val))


# ## Multi Target Model

# In[382]:

traincsv = pd.read_csv("/Users/jatingarg/Desktop/CompBioData/project1/p1_train_pop_lab.csv",low_memory=False)
accessionList = traincsv['accession'].values
sequencingCenterList = np.array(traincsv['sequencing_center'].values)
sequencingCenterList = sequencingCenterList.astype(str)
countryList = traincsv['population'].values


# In[383]:

dfTest1["SequencingCenter"] = sequencingCenterList


# In[384]:

dfTest1["Population"] = countryList


# In[385]:

dfTest1.head(2)


# In[386]:

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression


# In[391]:

cross_val = 5
averageAccuracyScore = 0
f1Pop,f1Seq = 0,0
rf_model_M = RandomForestClassifier(n_estimators=100, # Number of trees
                                  max_features=dfTest1.shape[1]-10,
                                    oob_score=False)    # Use OOB scoring*
# rf_model = LogisticRegression(penalty='l2')
multi_target_forest = MultiOutputClassifier(rf_model_M, n_jobs=-1)
for i in range(cross_val):
    msk = np.random.rand(len(dfTest1)) < 0.80

    train = dfTest1[msk]
    
    test = dfTest1[~msk]

    trainList = dfTest1.columns
    trainList = trainList[0:-1]

    features = trainList
    
    # Train the model
    multi_target_forest.fit(train.filter(dfTest1.columns[:-2]), train.filter(dfTest1.columns[-2:]))

    predictedDF = multi_target_forest.predict(test.filter(dfTest1.columns[:-2]))
    ansListSeq = test['SequencingCenter'].tolist()
    ansListPop = test['Population'].tolist()

    averageAccuracyScore += printAccuracy(ansListSeq,ansListPop,predictedDF)
    f1Pop += fScoreMulti(ansListPop,predictedDF[:,1:2].flatten())
    f1Seq += fScoreMulti(ansListSeq,predictedDF[:,0:1].flatten())

    print(averageAccuracyScore)
print("F1-Score For Population: ",f1Pop/float(cross_val))
print("F1-Score For Sequencing Center: ",f1Seq/float(cross_val))
print("Average Accuracy of model is " + str(averageAccuracyScore/cross_val))


# In[388]:

def printAccuracy(ansListSeq,ansListPop,predictedDF):
    count = 0
    for i in range(len(ansListSeq)):
        if ansListSeq[i] == predictedDF[i][0] and ansListPop[i] == predictedDF[i][1]:
            count += 1
#     print(countSeq/len(ansListSeq),countPop/len(ansListSeq))
    return count/float(len(ansListSeq))


# In[389]:

def fScoreMulti(y_true,y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred,average='micro')


# In[432]:

# save the model to disk as pickle files
filename = '/Users/jatingarg/Desktop/CompBioData/project1/multi_target_forest.pickle'
pickle.dump(multi_target_forest, open(filename, 'wb'))


# In[433]:

filename = '/Users/jatingarg/Desktop/CompBioData/project1/rf_modelSeq.pickle'
pickle.dump(rf_modelSeq, open(filename, 'wb'))


# In[434]:

filename = '/Users/jatingarg/Desktop/CompBioData/project1/rf_modelPop.pickle'
pickle.dump(rf_modelPop, open(filename, 'wb'))


# ### Finish
