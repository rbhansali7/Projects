
# coding: utf-8

# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import sys
import os
import pickle
get_ipython().magic('matplotlib inline')


# ### Here we are building the model using only the quant.sf file.

# In[3]:

df = pd.read_csv("project1/train/ERR188021/bias/quant.sf",low_memory=False,sep='\t')


# In[4]:

df.head(10)


# In[5]:

df.shape


# In[6]:

df.sort_values('TPM',ascending = False).head(10)


# In[7]:

df = df.filter(['Name','TPM'])


# ### 980 transcripts have more than 100 TPM count in 1 person. 

# In[8]:

df = df[df['TPM'] >= 100]
df.shape


# In[9]:

dfT = df.set_index("Name")
dfT = dfT.T


# In[10]:

transcriptList = dfT.columns


# In[11]:

dfT.columns


# In[12]:

transcriptDict = dict()
for i in range(len(transcriptList)):
    if transcriptDict.__contains__(transcriptList[i]):
        transcriptDict[transcriptList[i]] = transcriptDict.get(transcriptList[i]) + 1
    else:
        transcriptDict[transcriptList[i]] = 1


# In[7]:

traincsv = pd.read_csv("project1/p1_train.csv",low_memory=False)


# In[8]:

traincsv.head(10)


# In[9]:

traincsv['label'].value_counts()


# In[10]:

accessionList = traincsv['accession'].values
countryList = traincsv['label'].values


# In[11]:

transcriptDict = dict()


# ### This is the code to read quant.sf files for each person and here we are making superset of of all the transcripts having less than 100 TPM in any of the person.

# In[12]:

globaldf = pd.DataFrame()
def readAllQuantSFFiles():
    for i in range(len(accessionList)):
        name = accessionList[i]
        df = pd.read_csv("project1/train/"+ str(name) + "/bias/quant.sf",low_memory=False,sep='\t')
        extractTopTranscripts(df)


# In[13]:

def extractTopTranscripts(df):
    df = df.filter(['Name','TPM'])
    df = df[df['TPM'] <= 100]
    dfT = df.set_index("Name")
    dfT = dfT.T
    transcriptList = dfT.columns
    buildTranscriptDict(transcriptList)


# In[14]:

def buildTranscriptDict(transcriptList):
    for i in range(len(transcriptList)):
        global transcriptDict
        if transcriptDict.__contains__(transcriptList[i]):
            transcriptDict[transcriptList[i]] = transcriptDict.get(transcriptList[i]) + 1
        else:
            transcriptDict[transcriptList[i]] = 1


# In[15]:

readAllQuantSFFiles()


# In[16]:

print(len(transcriptDict))


# In[17]:

for key in transcriptDict.keys():
    print(key,transcriptDict.get(key))


# In[18]:

globaldf = pd.DataFrame()


# In[19]:

def makeGlobalDF():
    for i in range(len(accessionList)):
        name = accessionList[i]
        df = pd.read_csv("project1/train/"+ str(name) + "/bias/quant.sf",low_memory=False,sep='\t')
        addPersonCountry(df,i)     


# In[20]:

def addPersonCountry(df,index):
    df = df.filter(['Name','TPM'])
    dfT = df.set_index("Name")
    dfT = dfT.T
    dfT = dfT.filter(transcriptDict.keys())
    dfT['PersonID'] = accessionList[index]
    dfT['Country'] = countryList[index]
    global globaldf
    globaldf = globaldf.append(dfT)


# In[21]:

makeGlobalDF()


# In[22]:

globaldf.shape


# In[23]:

globaldf.head(10)


# In[ ]:

globaldf.to_csv("Global.csv",index=False)


# In[435]:

rf_model = RandomForestClassifier(n_estimators=500, # Number of trees
                                  max_features=1000,    # Num features considered
                                  oob_score=True)    # Use OOB scoring*
trainList = globaldf.columns
trainList = trainList[0:-3]

features = trainList
traindf = globaldf.head(300)
# Train the model
rf_model.fit(X=traindf[features],
             y=traindf["Country"])

print("OOB accuracy: ")
print(rf_model.oob_score_)


# In[436]:

testDF = globaldf.iloc[300:400]
predictedDF = rf_model.predict(testDF[features])


# In[437]:

predictedDF


# In[438]:

ansList = testDF['Country'].tolist()


# In[439]:

checkDict = dict()
for i in range(len(ansList)):
        global checkDict
        if checkDict.__contains__(ansList[i]):
            checkDict[ansList[i]] = checkDict.get(ansList[i]) + 1
        else:
            checkDict[ansList[i]] = 1


# In[440]:

print(checkDict)


# In[441]:

count = 0
for i in range(len(ansList)):
    if ansList[i] == predictedDF[i]:
        count += 1
print(count/len(ansList))


# In[32]:

confusion_matrix(ansList,predictedDF,labels=['GBR', 'FIN', 'CEU', 'TSI', 'YRI'])


# In[443]:

maskdf = pd.read_csv("Global.csv")


# In[28]:

maskdf = globaldf


# In[29]:

maskdf.shape


# ### Here we are training our model on transcripts having less than 100 TPM and we are getting 89.5 % accuracy.

# In[31]:

cross_val = 1
averageAccuracyScore = 0
rf_model = RandomForestClassifier(n_estimators=300, # Number of trees
                                    max_features=199000,    # Num features considered
                                    oob_score=False)    # Use OOB scoring*
for i in range(cross_val):
    msk = np.random.rand(len(maskdf)) < 0.80

    train = maskdf[msk]
    
    test = maskdf[~msk]


    trainList = maskdf.columns
    trainList = trainList[0:-3]

    features = trainList
    
#     XTrain_std = StandardScaler().fit_transform(train.filter(trainList))
#     XTest_std = StandardScaler().fit_transform(test.filter(trainList))

    # Train the model
    rf_model.fit(X=train[features],
                 y=train["Country"])
    
#     print("OOB accuracy: ")
#     print(rf_model.oob_score_)

    predictedDF = rf_model.predict(test[features])
    ansList = test['Country'].tolist()
    global averageAccuracyScore
    averageAccuracyScore += printAccuracy(ansList,predictedDF)
print("Average Accuracy of model is " + str(averageAccuracyScore/cross_val))


# In[26]:

def printAccuracy(ansList,predictedDF):
    count = 0
    for i in range(len(ansList)):
        if ansList[i] == predictedDF[i]:
            count += 1
#     print(count/len(ansList))
    return (count/len(ansList))


# In[72]:

def permutationTest():
    trainList = maskdf.columns
    trainList = trainList[0:-3]
    features = trainList
    predictedDF = rf_model.predict(maskdf[features])
    ansList = maskdf['Country'].tolist()
    permutationList = list()
    for i in range(500):
        dfshuffle = np.random.permutation(maskdf.Country)
        permutationList.append(printAccuracy(dfshuffle,predictedDF))
    print("Average Accuracy of permutation Test is :",sum(permutationList)/500)


# ### We have performed Permutation Test to check that our model is just not guessing the classes. The accuracy obtained on P test is just 20 % which shows our model is performing well.

# In[73]:

permutationTest()


# In[243]:

colList = globaldf.columns
colListWithoutCoun = colList[0:-2]
X_std = globaldf.filter(colListWithoutCoun)


# In[244]:

X_std.head(10)


# In[368]:

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# In[369]:

cov_mat.shape


# In[370]:

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[395]:

eig_vals[2000]


# In[375]:

Eig_List = list()
for i in range(len(eig_vals)):
    Eig_List.append(np.abs(eig_vals[i]))


# In[396]:

Eig_List[2000]


# In[380]:

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')

# for i in eig_pairs:
#     print(i[0])


# In[390]:

eig_pairs[2000][0]


# In[433]:

i = 0
indexList = list()
while i < 2000:
    for j in range(len(Eig_List)):
        if eig_pairs[i][0] == Eig_List[j]:
#             print(eig_pairs[i][0],Eig_List[j], j)
            indexList.append(j)
    i += 1


# In[434]:

len(indexList)


# In[424]:

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[425]:

colList = globaldf.columns


# In[426]:

filterList = list()
for i in range(len(indexList)):
    for j in range(len(colList)):
        if indexList[i] == j:
            filterList.append(colList[j])


# In[427]:

maskdf = globaldf.filter(filterList)


# In[428]:

maskdf['Country'] = globaldf["Country"].values


# In[429]:

maskdf.head(1)


# In[430]:

maskdf.shape


# ### By doing PCA using Eigen vectors and values the accuracy that we got is around 72.5 %

# In[432]:

cross_val = 5
averageAccuracyScore = 0
rf_model = RandomForestClassifier(n_estimators=1500, # Number of trees
                                    max_features=1581,    # Num features considered
                                    oob_score=False)    # Use OOB scoring*
for i in range(cross_val):
    msk = np.random.rand(len(maskdf)) < 0.80

    train = maskdf[msk]
    
    test = maskdf[~msk]


    trainList = maskdf.columns
    trainList = trainList[0:-1]

    features = trainList
    
#     XTrain_std = StandardScaler().fit_transform(train.filter(trainList))
#     XTest_std = StandardScaler().fit_transform(test.filter(trainList))

    # Train the model
    rf_model.fit(X=train[features],
                 y=train["Country"])
    
#     print("OOB accuracy: ")
#     print(rf_model.oob_score_)

    predictedDF = rf_model.predict(test[features])
    ansList = test['Country'].tolist()
    global averageAccuracyScore
    averageAccuracyScore += printAccuracy(ansList,predictedDF)
print("Average Accuracy of model is " + str(averageAccuracyScore/cross_val))


# In[464]:

with open('eqClasses_dict.pickle', 'wb') as handle:
    pickle.dump(eqClasses_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[465]:

with open('eqClasses_dict.pickle', 'rb') as handle:
    eqClasses_dict = pickle.load(handle)


# In[ ]:

eqDf = pd.DataFrame.from_dict(eqClasses_dict)


# In[ ]:

eqDf = pd.read_csv("eq.csv",low_memory=False)


# In[ ]:



