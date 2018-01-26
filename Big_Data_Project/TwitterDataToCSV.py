
# coding: utf-8

# In[46]:

import csv
import json
import io


# In[47]:

#Convert the json format twitter data dump into CSV file. Extract the relevant fields from it.

data_json = open('abc.json', mode='r').read() #reads in the JSON file into Python as a string
data_python = json.loads(data_json) #turns the string into a json Python object

csv_out = open('tweets_out_ASCII.csv', mode='w') #opens csv file
writer = csv.writer(csv_out) #create the csv writer object

fields = ['created_at', 'id_str','text', 'user-id_str', 'user-name', 'user-screen_name',
          'user-location', 'user-url', 'user-description', 'user-protected', 'user-verified',
          'user-followers_count', 'user-friends_count', 'user-listed_count', 'user-favourites_count',
          'user-statuses_count', 'user-created_at', 'user-utc_offset', 'user-time_zone', 'user-geo_enabled',
          'user-lang', 'user-following', 'geo', 'coordinates', 'place', 'contributors', 'is_quote_status',
          'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'favorited', 'retweeted',
          'filter_level', 'lang',] #field names
writer.writerow(fields) #writes field

for line in data_python:

    #writes a row and gets the fields from the json object
    #screen_name and followers/friends are found on the second level hence two get methods
    writer.writerow([line.get('created_at'),
                     line.get('id_str'),
                     line.get('text').encode('unicode_escape'), #unicode escape to fix emoji issue
                     line.get('user').get('id_str'),
                     line.get('user').get('name'),
                     line.get('user').get('screen_name'),
                     line.get('user').get('location'),
                     line.get('user').get('url'),
                     line.get('user').get('description'),
                     line.get('user').get('protected'),
                     line.get('user').get('verified'),
                     line.get('user').get('followers_count'),
                     line.get('user').get('friends_count'),
                     line.get('user').get('listed_count'),
                     line.get('user').get('favourites_count'),
                     line.get('user').get('statuses_count'),
                     line.get('user').get('created_at'),
                     line.get('user').get('utc_offset'),
                     line.get('user').get('time_zone'),
                     line.get('user').get('geo_enabled'),
                     line.get('user').get('lang'),
                     line.get('user').get('following'),
                     line.get('geo'),
                     line.get('coordinates'),
                     line.get('place'),
                     line.get('contributors'),
                     line.get('is_quote_status'),
                     line.get('quote_count'),
                     line.get('reply_count'),
                     line.get('retweet_count'),
                     line.get('favorite_count'),
                     line.get('favorited'),
                     line.get('retweeted'),
                     line.get('filter_level'),
                     line.get('lang'),])

csv_out.close()


# In[4]:

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import sys
#from tweepy import api
import csv
import tweepy

access_key = "2264333875-7wNO0rdlU7SporFM6wWrheLBALn6St4vRUmYlLK"
access_secret = "SOpxEyM3AAM39ubfmMBempXY8JtstEl5yLm7DqVlRZNhT"
consumer_key = "Bas2wIPWSjVyWg0A4ltXgjvb0"
consumer_secret = "5ge6FhICgxCQVjXnrhBzgvSwDxZW8aUooDPhPOGnaZFJAcH2jH"

#use variables to access twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


#create an object called 'customStreamListener'
class CustomStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        
        fields = ['created_at', 'id_str','SentimentText', 'user-id_str', 'user-name', 'user-screen_name',
          'user-location', 'user-url', 'user-description', 'user-protected', 'user-verified',
          'user-followers_count', 'user-friends_count', 'user-listed_count', 'user-favourites_count',
          'user-statuses_count', 'user-created_at', 'user-utc_offset', 'user-time_zone', 'user-geo_enabled',
          'user-lang', 'user-following', 'geo', 'coordinates', 'place', 'contributors', 'is_quote_status',
          'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'favorited', 'retweeted',
          'filter_level', 'lang',] #field names
        
        with open('rawData.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([status.created_at,status.id_str,status.text,status.user.id_str,status.user.name,
                             status.user.screen_name,status.user.location,status.user.url,status.user.description,
                            status.user.protected,status.user.verified,status.user.followers_count,
                            status.user.friends_count,status.user.listed_count,status.user.favourites_count,
                            status.user.statuses_count,status.user.created_at,status.user.utc_offset,
                            status.user.time_zone,status.user.geo_enabled,status.user.lang,status.user.following,
                            status.geo, status.coordinates, status.place, status.contributors,status.is_quote_status,
                            status.quote_count,status.reply_count,status.retweet_count,status.favorite_count,
                            status.favorited,status.retweeted,status.filter_level,status.lang])


    def on_error(self, status_code):
        return True

    def on_timeout(self):
        return True

streamingAPI = tweepy.streaming.Stream(auth, CustomStreamListener())
streamingAPI.filter(track=['woman', 'women', 'girl', 'girls'])

