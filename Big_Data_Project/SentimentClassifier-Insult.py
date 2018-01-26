
# coding: utf-8

# In[103]:

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[8]:

def dataShaper():
    data = pd.read_csv('insult_train.csv', encoding = "ISO-8859-1")
    data2 = pd.read_csv('xaa.csv', encoding = "ISO-8859-1")
    data3 = pd.read_csv('xab.csv', encoding = "ISO-8859-1")
    data4 = pd.read_csv('xac.csv', encoding = "ISO-8859-1")
    data5 = pd.read_csv('xad.csv', encoding = "ISO-8859-1")
    data6 = pd.read_csv('xae.csv', encoding = "ISO-8859-1")
    data7 = pd.read_csv('xaf.csv', encoding = "ISO-8859-1")
    data8 = pd.read_csv('xag.csv', encoding = "ISO-8859-1")
    data9 = pd.read_csv('xah.csv', encoding = "ISO-8859-1")
    data10 = pd.read_csv('violent_tweets_sentiment_1.csv', encoding = "ISO-8859-1")
    data11 = pd.read_csv('training_1_sentiment_0.csv', encoding = "ISO-8859-1")
    data12 = pd.read_csv('training_2_sentiment_0.csv', encoding = "ISO-8859-1")
    
    data13 = pd.read_csv('tweets.csv', encoding = "ISO-8859-1")
    data13 = data13[data13['Sentiment']==4]
    data13['Sentiment'] = data13['Sentiment'].map( {4:0} )
    
    
    
    data = data.append(data2)
    data = data.append(data3)
    data = data.append(data4)
    data = data.append(data5)
    data = data.append(data6)
    data = data.append(data7)
    data = data.append(data8)
    data = data.append(data9)
    data = data.append(data10)
    data = data.append(data11)
    data = data.append(data12)
    data = data.append(data13)
    
    #remove extra columns
    data = data.filter(['Sentiment','SentimentText'])
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map(int)
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    return data

data = dataShaper()
data.head(5)


# In[9]:

#Make classifier
def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'NC'


# In[10]:

def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['SentimentText'].progress_map(tokenize)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


# In[11]:

data = postprocess(data)


# In[12]:

data.head(5)


# In[13]:

n=1000000
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).Sentiment), test_size=0.2)


# In[14]:

x_train[0]


# In[15]:

n_dim=200
token_count = sum([len(sentence) for sentence in x_train])
tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab(x_train)
tweet_w2v.train(x_train, total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


# In[19]:

tweet_w2v.most_similar('facebook')


# In[20]:

print ('Mkae tf-idf matrix:')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(x_train)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('vocab size :', len(tfidf))


# In[21]:

#Now let's define a function that, given a list of tweet tokens, creates an averaged tweet vector.
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


# In[22]:

from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs_w2v = scale(test_vecs_w2v)


# In[23]:

from keras.models import Sequential
from keras.layers import Activation, Dense
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)


# In[24]:

score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print (score[1])


# In[25]:

#for checking on test data
def dataShaper2(fileName):
    data = pd.read_csv(fileName, encoding = "ISO-8859-1")
    data = data.filter(['Sentiment','SentimentText'])
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map(int)
    data = data[np.logical_or(data['Sentiment']==4,data['Sentiment']==0)]
    data['Sentiment'] = data['Sentiment'].map( {0:1, 4:0} )
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', data.shape)   
    return data

fileName = 'testdata.manual.2009.06.14.csv'
data2 = dataShaper2(fileName)
data2.head(5)


# In[26]:

testData = postprocess(data2)


# In[36]:

n=1000000
x_test_train, x_test, y_test_train, y_test = train_test_split(np.array(testData.head(n).tokens),
                                                    np.array(testData.head(n).Sentiment), test_size=0.0)


# In[37]:

test_train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_test_train])
test_train_vecs_w2v = scale(test_train_vecs_w2v)


# In[38]:

#Accuracy on test data
score = model.evaluate(test_train_vecs_w2v, y_test_train, batch_size=128, verbose=2)
print (score[1])


# In[39]:

#model.predict()   <- returns a list of output


# In[40]:

fields = ['created_at', 'id_str','SentimentText', 'user-id_str', 'user-name', 'user-screen_name',
          'user-location', 'user-url', 'user-description', 'user-protected', 'user-verified',
          'user-followers_count', 'user-friends_count', 'user-listed_count', 'user-favourites_count',
          'user-statuses_count', 'user-created_at', 'user-utc_offset', 'user-time_zone', 'user-geo_enabled',
          'user-lang', 'user-following', 'geo', 'coordinates', 'place', 'contributors', 'is_quote_status',
          'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'favorited', 'retweeted',
          'filter_level', 'lang',] 


# In[41]:

def labelPredict(fileName):
    testVec = makeTestDataFrame(fileName)
    testData = postprocess(testVec)
    n=testData.shape[0]
    testArray = np.array(testData)
#     x_test_train, x_test, y_test_train, y_test = train_test_split(np.array(testData.head(n).tokens),
#                                                     np.array(testData.head(n).Sentiment), test_size=0.0)
    n_dim = 200
    testArray = np.array(testData.head(n).tokens)
    test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in testArray])
    test_vecs_w2v = scale(test_vecs_w2v)
    return model.predict(test_vecs_w2v), model.predict_classes(test_vecs_w2v)


# In[42]:

#for checking on test data
def makeTestDataFrame(fileName):
    global fields
    data = pd.read_csv(fileName, encoding = "ISO-8859-1")                      
    data.columns = fields                        
    data = data.filter(['SentimentText'])
    
#     data = data[data.Sentiment.isnull() == False]
#     data['Sentiment'] = data['Sentiment'].map(int)
#     data = data[np.logical_or(data['Sentiment']==4,data['Sentiment']==0)]
#     data['Sentiment'] = data['Sentiment'].map( {0:1, 4:0} )
#     #data['Sentiment'] = data['Sentiment'].map( {4:1, 0:0} )
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', data.shape)   
    return data


# In[43]:

fileName = "rawData.csv"
predictedProbDF, predictedLabels = labelPredict(fileName)


# In[44]:

testVec = makeTestDataFrame(fileName)


# In[45]:

testVec.head(10)


# In[881]:

# Graph showing the intensity of tweets


# In[166]:

intensityPlot(predictedProbDF)


# In[161]:

def intensityPlot(predictedProbDF):
    predictedProbDFList = np.array(predictedProbDF)
    plt.hist(predictedProbDFList,300, facecolor='blue', alpha=0.75)
    plt.xlabel('Tweet Intensity')
    plt.ylabel('Number of Tweets')
    plt.rcParams.update({'font.size': 15})
    fig = plt.gcf()
    fig.set_size_inches(10.5,5.5)


# In[882]:

# CDF of the above graph


# In[168]:

intensityCumPlot(predictedProbDF)


# In[167]:

def intensityCumPlot(predictedProbDF):
    unique, counts = np.unique(predictedProbDF, return_counts=True)
    counts = np.cumsum(counts)
    plt.plot(unique,counts, alpha=0.75)
    plt.xlabel('Tweet Intensity')
    plt.ylabel('Number of Tweets')
    plt.rcParams.update({'font.size': 15})
    fig = plt.gcf()
    fig.set_size_inches(10.5,5.5)


# In[192]:

maxi = 0.98


# In[88]:

len(predictedProbDF)


# In[145]:

testVec["PredictedProb"] = predictedProbDF


# In[146]:

testVec["PredictedClass"] = predictedLabels


# In[147]:

testVec.head(10)


# In[157]:

testVectLevel1 = testVec.copy()


# In[158]:

testVectLevel1 = testVectLevel1[np.logical_and(testVec['PredictedProb'] >= 0.7, testVec['PredictedProb'] < 0.8)]


# In[883]:

# Severity Level Wise Distribution of tweets.


# In[169]:

intensityPlot(testVectLevel1['PredictedProb'])


# In[171]:

testVectLevel2 = testVec[np.logical_and(testVec['PredictedProb'] >= 0.8, testVec['PredictedProb'] < 0.9)]
intensityPlot(testVectLevel2['PredictedProb'])


# In[193]:

testVectLevel3 = testVec[np.logical_and(testVec['PredictedProb'] >= 0.9, testVec['PredictedProb'] < maxi)]
intensityPlot(testVectLevel3['PredictedProb'])


# In[176]:

print("Percentage of tweets with Level 1 intensity", str((testVectLevel1.shape[0]*100)/float(testVec.shape[0]))[0:4])


# In[177]:

print("Percentage of tweets with Level 2 intensity", str((testVectLevel2.shape[0]*100)/float(testVec.shape[0]))[0:4])


# In[194]:

print("Percentage of tweets with Level 3 intensity", str((testVectLevel3.shape[0]*100)/float(testVec.shape[0]))[0:4])


# In[800]:

len(testVec)


# In[195]:

data = pd.read_csv(fileName, encoding = "ISO-8859-1")                      
data.columns = fields                        
location = data['user-location']


# In[197]:

testVec['user-location'] = location


# In[375]:

testVec.head(10)


# In[628]:

testVec.shape


# In[629]:

testVecLoc = testVec.copy()


# In[630]:

testVecLoc = testVecLoc.dropna()


# In[631]:

testVecLoc.shape[0]


# In[632]:

# testVecLoc['user-location'] = testVecLoc['user-location'].apply(lambda x: x.encode('utf-8').strip())


# In[633]:

# testVecLoc['user-location'] = testVecLoc['user-location'].apply(lambda x: str(x))


# In[712]:

testVectLevel4 = testVecLoc[np.logical_and(testVec['PredictedProb'] >= 0.7, testVec['PredictedProb'] <= 1.0)]


# In[713]:

testVectLevel4.head(10)


# In[714]:

testVecLocNp = np.array(testVectLevel4)


# In[715]:

import string


# In[703]:

def removePunctuations(s):
    newS = s
    for char in string.punctuation:
        newS = newS.replace(char, " ")
    output = [s.strip().lower() for s in newS.split() if s]
    return output


# In[704]:

def getWord(line):
    validwd=[]
#     line = line.decode()
    line = removePunctuations(line)
    for wd in line:
        wd = str(wd)
        if(wd.isalpha()):
            validwd.append(wd)
    return " ".join(validwd)


# In[609]:

worldDict = dict()


# In[610]:

with open('worldcitiespop.txt',mode='r',encoding='ISO-8859-1') as file:
    for line in file:
        lis = line.split(",")
        worldDict[lis[1].strip().lower()] = lis[1].strip().lower()
        worldDict[lis[2].strip().lower()] = lis[2].strip().lower()


# In[716]:

len(worldDict)


# In[717]:

c = 0
for i in range(testVecLocNp.shape[0]):
    word = getWord(testVecLocNp[i][3])
    wordLis = word.split(" ")
    flag = 0
    for word in wordLis:
        if worldDict.__contains__(word) == True:
            testVecLocNp[i,3] = word
            flag = 1
            break
    if flag == 0:
        testVecLocNp[i,3] = np.NaN


# In[718]:

testVecLocNp = pd.DataFrame(testVecLocNp,columns=set(testVectLevel4.columns))


# In[719]:

testVecLocNp.head(2)


# In[720]:

testVecLocNp.isnull().sum()


# In[721]:

testVecLocNpNA = testVecLocNp.dropna()


# In[723]:

testVecLocNpNA.shape


# In[724]:

testVecLocNpNA.head(10)


# In[725]:

testVecLocNpList = testVecLocNpNA['user-location']


# In[726]:

type(testVecLocNpList)


# In[727]:

testVecLocNpList = np.array(testVecLocNpList)
len(testVecLocNpList)


# In[728]:

unique, counts = np.unique(testVecLocNpList, return_counts=True)


# In[729]:

mapArray = list()
for i in range(len(unique)):
    mapArray.append([unique[i],counts[i]])


# In[730]:

len(mapArray)


# In[749]:

cityLatLongDF = pd.read_csv('citiesTolatlong.csv',encoding='ISO-8859-1')


# In[751]:

cityLatLongDF.head(10)


# In[752]:

cityLatLongDF = cityLatLongDF.drop_duplicates(subset='City')


# In[753]:

cityLatLongDF.shape


# In[754]:

cityLatLongDF = np.array(cityLatLongDF)


# In[762]:

latLongDict = dict()
for i in tqdm(range(cityLatLongDF.shape[0])):
    latLongDict[cityLatLongDF[i][0]] = tuple([cityLatLongDF[i][2],cityLatLongDF[i][1]])


# In[763]:

len(latLongDict)


# In[731]:

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
import math


# In[764]:

cities = mapArray
scale = 0.1
plt.figure(figsize=(20,10))
map = Basemap()

map.drawmapboundary(fill_color='skyblue')
map.fillcontinents(color='black',lake_color='skyblue')

# load the shapefile, use the name 'states'
map.readshapefile('/Users/jatingarg/Downloads/world/world_countries_boundary_file_world_2002', name='countries', drawbounds=True)

# Get the location of each city and plot it
geolocator = Nominatim()
c = 0
for index in tqdm(range(len(cities))):
    city,count=cities[index]
    if latLongDict.__contains__(city) == False:
        c += 1
        continue
#     try:
#         loc = geolocator.geocode(city)
#         if(loc==None):
#             continue
#     except Exception as e: 
#         print(e)

#     print("city",city," longitude",loc.longitude," latitude",loc.latitude)    
    x, y = map(latLongDict.get(city)[0], latLongDict.get(city)[1])
    map.plot(x,y,marker='o',color='White',markersize=int(math.sqrt(count))*scale)
plt.show()


# In[803]:

mapArray.sort(key=lambda x:x[1],reverse=True)
cities = mapArray[:]
scale = 0.1
plt.figure(figsize=(20,10))
map = Basemap()
# map = Basemap(width=12000000,height=9000000,projection='lcc',
#             resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)

# map.drawmapboundary(fill_color='skyblue')

map.fillcontinents(color='black',lake_color='darkblue')

# load the shapefile, use the name 'states'
map.readshapefile('/Users/jatingarg/Downloads/world/world_countries_boundary_file_world_2002', name='countries', drawbounds=True)
map.bluemarble()
# Get the location of each city and plot it
geolocator = Nominatim()
c = 0
for index in tqdm(range(len(cities))):
    city,count=cities[index]
    if latLongDict.__contains__(city) == False:
        c += 1
        continue
#     try:
#         loc = geolocator.geocode(city)
#         if(loc==None):
#             continue
#     except Exception as e: 
#         print(e)

#     print("city",city," longitude",loc.longitude," latitude",loc.latitude)    
    x, y = map(latLongDict.get(city)[0], latLongDict.get(city)[1])
    map.plot(x,y,marker='o',color='White',markersize=int(math.sqrt(count))*scale)
plt.savefig('worldMap.png')
plt.show()


# In[771]:

cities = mapArray.sort(key=lambda x:x[1],reverse=True)


# In[812]:

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='once')


# In[843]:

def makeUserDF(fileName):
    user = pd.read_csv(fileName,error_bad_lines=False,sep=';')
    user = user.filter(['text'])
    user.columns = ["SentimentText"]
    user = user[user['SentimentText'].isnull() == False]
    user.reset_index(inplace=True)
    user.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', user.shape)   
    return user


# In[836]:

from sklearn.preprocessing import scale


# In[839]:

def labelPredictUser(testVec):
    testData = postprocess(testVec)
    n=testData.shape[0]
    testArray = np.array(testData)

    n_dim = 200
    testArray = np.array(testData.head(n).tokens)
    test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in testArray])
    
    test_vecs_w2v = scale(test_vecs_w2v)
    return model.predict(test_vecs_w2v), model.predict_classes(test_vecs_w2v)


# In[868]:

def stats(name,fileName):
    user = makeUserDF(fileName)
    predProb, predLabel = labelPredictUser(user)
    val = np.sum(predProb)/float(len(predProb))
    st = ""
    if val<0.7:
        st = " Neutral"
    elif val<0.8:
        st = " Level 1"
    elif val<0.9:
        st = " Level 2"
    else:
        st = ' Level 3'
    print(name + " intensity score : " + str(val) + st)


# In[869]:

fileName = 'obama.csv'
stats('obama',fileName)


# In[871]:

fileName = 'trump.csv'
stats('trump',fileName)


# In[872]:

fileName = 'popefrancis.csv'
stats("popefrancis",fileName)


# In[873]:

fileName = 'morgan.csv'
stats('morgan',fileName)


# In[874]:

fileName = 'krk.csv'
stats('krk',fileName)


# ## Finish
