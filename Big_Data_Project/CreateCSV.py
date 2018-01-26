from pyspark import SparkContext, SparkConf

# Creating the spark configuration and SparkContext
conf = SparkConf().setAppName("CreateCSV").setMaster("local")
sc = SparkContext(conf=conf)

import json
import csv


def chunkify(input):
    if isinstance(input, dict):
        return {chunkify(key): chunkify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [chunkify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

dictRdd = sc.textFile("Dummy_Data.txt").filter(lambda x : x != '')

dictList = dictRdd.map(lambda x : chunkify(json.loads(x))).collect()


keys = dictList[0].keys()
with open('Final.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys, extrasaction='ignore')
    dict_writer.writeheader()
    dict_writer.writerows(dictList)

