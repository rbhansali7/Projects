
import sys
import pandas as pd
import numpy as np
from scipy import linalg
import pickle
import sklearn.metrics


eqClasses_dict = {}

def storeInDict(line, person, count):
    s = line.split()
    n = len(s)
    eqClassLength = int(s[0])
    tup = list()

    # Storing length 1 equivalence class as a string instead of a tuple in the dictionary
    if (eqClassLength == 1):
        t_name = s[1]
        if t_name not in eqClasses_dict:
            eqClasses_dict[t_name] = [0 for i in range(count)]
            eqClasses_dict[t_name][person] = int(s[n - 1])
        else:
            eqClasses_dict[t_name][person] = int(s[n - 1])

    else:
        # Create the tuple of transcript IDs which will be the key
        for i in range(1, eqClassLength + 1):
            tup.append(s[i])
        tup = tuple(tup)

        if tup not in eqClasses_dict:
            eqClasses_dict[tup] = [0 for i in range(count)]
            eqClasses_dict[tup][person] = int(s[n - 1])
        else:
            eqClasses_dict[tup][person] = int(s[n - 1])


def parseInput(path, personCount, count):
    file = open(path, "r")

    lineCount = 0
    num_transcripts = 0
    for line in file:
        if lineCount == 0:
            num_transcripts = int(line)
            lineCount += 1
            continue
        if lineCount == 1:
            num_eqClasses = int(line)
            lineCount += 1
            continue
        if lineCount < 2 + num_transcripts:
            lineCount += 1
            continue
        else:
            storeInDict(line, personCount, count)
            lineCount += 1
            continue

def parseAllFiles(fileNamePredict, fileNameTest):
    traincsv = pd.read_csv(fileNameTest, low_memory=False)
    accessionList = traincsv['accession'].values

    count = len(accessionList)

    for i in range(len(accessionList)):
        name = accessionList[i]
        s = str(fileNamePredict) + str(name) + "/bias/aux_info/eq_classes.txt"
        parseInput(s, i, count)


def printAccuracy(ansListSeq, ansListPop, predictedDF):
    count = 0
    for i in range(len(ansListSeq)):
        if ansListSeq[i] == predictedDF[i][0] and ansListPop[i] == predictedDF[i][1]:
            count += 1
    return count / float(len(ansListSeq))

def fScoreMulti(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, average='micro')


def executor(filenameModel, fileNamePredict, fileNameTest):
    parseAllFiles(fileNamePredict, fileNameTest)
    df = pd.DataFrame.from_dict(eqClasses_dict)
    colList = df.columns
    readList = list()
    for i in range(df.shape[1]):
        num = (df[colList[i]] != 0).sum()
        tup = tuple([num, i])
        readList.append(tup)
    sortedList = list()
    sortedList = sorted(readList, key=lambda x: x[0])
    sortedNp = np.array(sortedList)
    sortedDF = pd.DataFrame(sortedNp)
    tempDF = sortedDF.copy()
    sortedDF = tempDF[tempDF[0] >= 350]
    sortedDF = sortedDF[sortedDF[0] <= 369]

    uniqueDF = sortedDF
    colIndex = uniqueDF[1].tolist()
    colNames = list()
    for i in range(len(colIndex)):
        colNames.append(colList[colIndex[i]])
    dfTest1 = df.filter(colNames)

    traincsv = pd.read_csv(fileNameTest, low_memory=False)
    accessionList = traincsv['accession'].values
    sequencingCenterList = np.array(traincsv['sequencing_center'].values)
    sequencingCenterList = sequencingCenterList.astype(str)
    countryList = traincsv['population'].values

    dfTest1["SequencingCenter"] = sequencingCenterList
    dfTest1["Population"] = countryList

    multi_target_forest = pickle.load(open(filenameModel, 'rb'))
    predictedDF = multi_target_forest.predict(dfTest1.filter(dfTest1.columns[:-2]))
    ansListSeq = dfTest1['SequencingCenter'].tolist()
    ansListPop = dfTest1['Population'].tolist()

    averageAccuracyScore = printAccuracy(ansListSeq, ansListPop, predictedDF)
    f1Pop = fScoreMulti(ansListPop, predictedDF[:, 1:2].flatten())
    f1Seq = fScoreMulti(ansListSeq, predictedDF[:, 0:1].flatten())

    print("F1-Score For Population: ", f1Pop)
    print("F1-Score For Sequencing Center: ", f1Seq)
    print("Average Accuracy of model is " + str(averageAccuracyScore))


# In[62]:

# executor(filenameModel,fileNamePredict,fileNameTest)


# In[59]:

# filenameModel = '/Users/jatingarg/Desktop/CompBioData/project1/multi_target_forest.pickle'


# In[60]:

# fileNameTest = '/Users/jatingarg/Desktop/CompBioData/project1/p1_train_pop_lab.csv'


# In[61]:

# fileNamePredict = '/Users/jatingarg/Desktop/CompBioData/project1/train/'


# In[64]:

def main():
    filenameModel = sys.argv[1]
    fileNamePredict = sys.argv[2]
    fileNameTest = sys.argv[3]
    executor(filenameModel, fileNamePredict, fileNameTest)


# In[ ]:

if __name__ == "__main__":
    main()