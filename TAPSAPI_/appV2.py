import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib
from matplotlib import pyplot as plt
import joblib
from flask import Flask, request, jsonify
import joblib
import json
import random

app = Flask(__name__)
MODEL = joblib.load('logisticReg_v1.pkl')
VECTORIZER = joblib.load('tfidVector_v2.pkl')

def getDay(data):
    #data['created_at']
    days={'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6,'Sun':7}
    return days[data.split(' ')[0]]
def getMonth(data):
    months={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    return months[data.split(' ')[1]]
def getTime(data):
    return data.split(' ')[3].split(':')[0]

stopword = ['itaas', 'kanyang', 'katulad', 'mismo', 'am', 'amin', 'ilalim',
       'ginagawa', 'kong', 'makita', 'tungkol', 'para', 'tayo', 'kapag',
       'paggawa', 'kanya', 'kulang', 'napaka', 'ginawang', 'marapat',
       'aking', 'bawat', 'laban', 'mayroon', 'bakit', 'iyon', 'tulad',
       'hanggang', 'ibig', 'kailangan', 'ilagay', 'muli', 'ito', 'ibabaw',
       'may', 'narito', 'sarili', 'pamamagitan', 'aming', 'mga', 'iyong',
       'sabihin', 'kanila', 'pangalawa', 'walang', 'atin', 'nila',
       'kanilang', 'marami', 'karamihan', 'nabanggit', 'likod', 'pareho',
       'gusto', 'maaari', 'kumuha', 'sa', 'namin', 'lima', 'anumang',
       'pa', 'ko', 'kung', 'mahusay', 'din', 'pababa', 'kaysa', 'hindi',
       'ni', 'kahit', 'sila', 'dalawa', 'dahil', 'nilang', 'niya',
       'habang', 'lahat', 'na', 'ng', 'maging', 'palabas', 'pagitan',
       'sabi', 'alin', 'ibaba', 'minsan', 'pagkakaroon', 'at', 'paraan',
       'ay', 'gayunman', 'ngayon', 'dito', 'katiyakan', 'siya', 'tatlo',
       'ako', 'ang', 'una', 'gumawa', 'niyang', 'nakita', 'isa', 'nasaan',
       'noon', 'pagkatapos', 'kami', 'iyo', 'gagawin', 'lamang', 'apat',
       'bago', 'naging', 'kanino', 'nito', 'paano', 'pero', 'ano', 'ka',
       'pataas', 'mula', 'ikaw', 'ginawa', 'o', 'maaaring', 'pumunta',
       'akin', 'nais', 'sino', 'saan', 'inyong', 'ating', 'kailanman',
       'bababa', 'bilang', 'huwag', 'isang', 'dapat', 'kapwa', 'iba',
       'masyado', 'pumupunta', 'panahon', 'nagkaroon', 'doon', 'kaya',
       'ilan','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
       "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
       'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
       'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
       'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
       'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
       'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
       'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
       'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
       'by', 'for', 'with', 'about', 'against', 'between', 'into',
       'through', 'during', 'before', 'after', 'above', 'below', 'to',
       'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
       'again', 'further', 'then', 'once', 'here', 'there', 'when',
       'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
       'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
       'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
       'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
       'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
       "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
       'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
       'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
       'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
       'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
sw = np.array(stopword)

def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

def set_random(cols):
    return int(random.random()*100)
def set_randomInt(cols):
    return random.randint(0,100)
def set_direction(text):
    if len(text.split(' ')[-1]) != 2:
        return 'na'
    else:
        return text.split(' ')[-1]
def vectorize(data,vectorizer):
    # extract the tfid representation matrix of the text data
    tfid_matrix = vectorizer.transform(data['full_text'])
    # collect the tfid matrix in numpy array
    array = tfid_matrix.todense()
    
    df = pd.DataFrame(array)
    data2 = data.iloc[:,4:].reset_index(drop=True)
    new_data=pd.concat((df,data2),axis=1)
    new_data.columns = new_data.columns.astype(str)
    return new_data

def find_location(data,label_col):
    #label_col - name of column that tags tweets
    #for 3&2 find WB, EB, SB, NB
    #for 1&0 - find GPE and/or Ave* *city (for GPE check if 1. proper noun, 2. if there are similar tags, 3. contains only 1 word)
    f32 = data[data[label_col].isin([2,3])]
    list32 = []
   
    for text in f32['full_text']:
        print(text)
        textList = str(text).split(' ')
        try:
            start_index = textList.index('at')
        except ValueError:
            #find other patterns
            continue
        for i in ['SB','NB','WB','EB']:
            try:
                end_index = textList.index(i)
                break
            except ValueError:
                continue
        #use comment for NOT adding nb, sb, wb, eb
        #list32.append(' '.join(textList[start_index+1:end_index]))
        list32.append(' '.join(textList[start_index+1:end_index+1]))
    edsa_locs = []
    nonEDSALocs = []
    for i in list32:
        if i.split(' ')[0] == 'EDSA':
            edsa_locs.append(i)
        else:
            nonEDSALocs.append(i)
    final_data = pd.DataFrame(edsa_locs,columns=['Location'])
    return final_data
@app.route('/predict_v1', methods=['GET'])
def predict():
    #uses twitter data
    #takes json type of data
    #takes from RapidAPI
    orig_data = pd.DataFrame(pd.read_csv('LastTestingFormat.csv'))
    data = pd.DataFrame(pd.read_csv('LastTestingFormat.csv'))
    data['date_days'] = data['created_at'].apply(getDay)
    data['date_months'] = data['created_at'].apply(getMonth)
    data['date_time'] = data['created_at'].apply(getTime)
    #----------------------------------------------------------#
    data['full_text'] = data['full_text'].apply(remove_punctuation)
    data['full_text'] = data['full_text'].apply(stopwords)
    new_data = vectorize(data,VECTORIZER)
    data['pred_tags'] = MODEL.predict(new_data)
    data['full_text'] = orig_data['full_text']
    #randomizer first for testing purposes\
    final_data = find_location(data,'pred_tags')
    final_data['accident_prediction'] = final_data['Location'].apply(set_random)
    final_data['total_accidents'] = final_data['Location'].apply(set_randomInt)
    final_data['direction'] = final_data['Location'].apply(set_direction)
    return json.loads(final_data.to_json(orient='index'))
   




if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)  # run our Flask app