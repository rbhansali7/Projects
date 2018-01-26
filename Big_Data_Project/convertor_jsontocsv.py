import unicodecsv as csv
import json
import io
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import os.path


def convert(infile,outfile):
    if(os.path.exists(outfile)):
        print "file already converted =>",infile," to ",outfile
        return
    print "converting infile =>",infile,", to outfile=>",outfile
    data_json = open(infile, mode='r').read() #reads in the JSON file into Python as a string
    data_python = json.loads(data_json) #turns the string into a json Python object

    csv_out = open(outfile, mode='w') #opens csv file
    writer = csv.writer(csv_out) #create the csv writer object

    fields = ['created_at', 'id_str','text', 'user-id_str', 'user-name', 'user-screen_name',
              'user-location', 'user-url', 'user-description', 'user-protected', 'user-verified',
              'user-followers_count', 'user-friends_count', 'user-listed_count', 'user-favourites_count',
              'user-statuses_count', 'user-created_at', 'user-utc_offset', 'user-time_zone', 'user-geo_enabled',
              'user-lang', 'user-following', 'geo', 'coordinates', 'place', 'contributors', 'is_quote_status',
              'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'favorited', 'retweeted',
              'filter_level', 'lang',] #field names
    writer.writerow(fields) #writes field
    fail=0
    for i in tqdm(range(0,len(data_python))):
    # for line in tqdm(data_python):
        line = data_python[i]
        #writes a row and gets the fields from the json object
        #screen_name and followers/friends are found on the second level hence two get methods
        try:
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
        except:
            fail+=1
            pass
    print "infile =>",infile," failed count =>",fail
    csv_out.close()   
    return fail 


def getArgs(argv):
    infile=None
    outfile=None
    try:
        opts, args = getopt.getopt(argv,"i:o:",["infile","outfile"])
    except getopt.GetoptError:
        print('JsonToCsvConvertor.py -i <infile> -o <outfile>') 
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-o", "--outfile"):
            outfile = arg
    return infile,outfile

def convertHandleException(infile,outfile):
    try:
        failRecord=convert(infile,outfile)
    except Exception e:
        print("infile=>",infile,"error occured =>",e)
        pass


def main(argv):

    infile,outfile=getArgs(argv)
    print "args: infile=>",infile,", outfile=>",outfile
    if(infile!=None):
        if(outfile==None):
            outfile=infile+".csv"
        convertHandleException(infile,outfile)
    else:
        files=['xaa','xab','xac','xad','xae','xaf','xag','xah']
        failedFiles=[]

        failedFile=''
        try:
            for fil in files:
                failedFile=fil
                print "converting file =>",fil
                infile=fil
                outfile=infile+".csv"
                convertHandleException(infile,outfile)
        except:
            failedFiles.append(failedFile)

        print "failedFiles =>",failedFiles


if __name__ == '__main__':
    main(sys.argv[1:])






