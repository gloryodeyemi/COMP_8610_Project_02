import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import contractions

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from tensorflow.keras import models
# from tensorflow.keras import layers
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
# from keras.utils import pad_sequences

def clean(data):   
    print("-"*13)
    print("Cleaning data")
    print("-"*13)
    
    # replace new line with space
    data['clean'] = data['comment_text'].apply(lambda row: re.sub("\n", " ", row))
    # replace non-characters with ""
    data['clean'] = data['clean'].apply(lambda text: re.sub("[^A-Za-z\' ]+", "", text))
    # replace non-characters with ""
    data['clean'] = data['clean'].apply(lambda text: re.sub("([^\x00-\x7F])+"," ",text))
    # replace numbers with ""
    data['clean'] = data['clean'].apply(lambda text: re.sub('[0-9]',"",text))
    # remove repeated text
    data['clean'] = data['clean'].apply(lambda text: re.sub(r'(.)\1{2,}', r'\1',text))
    # remove html tags
    data['clean'] = data['clean'].apply(lambda text: re.sub(r"<.*>"," ",text, flags=re.MULTILINE))
    # remove links
    data['clean'] = data['clean'].apply(lambda text: re.sub(r"http\S+"," ",text, flags=re.MULTILINE))

    # remove stopwords from comment
    stop_words = stopwords.words('english')

    data['clean'] = data['clean'].apply(
        lambda text: " ".join([word.lower() for word in text.split() if word not in stop_words]))
    print("Data cleaned!")
    
    return data

def decontraction(text):
    result=""
    for word in text.split():
        con_text=contractions.fix(word)
        if con_text.lower() is word.lower():
            result=result+word+" "
        else:
            result=result+con_text+" "

    return result.strip()

def decontract(data):
    print("-"*18)
    print("Decontracting text")
    print("-"*18)
    
    data['decontracted'] = data['clean'].apply(lambda text: decontraction(text))
    
    print("Texts decontracted!")
    
    return data

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    result = ""
    for word in text.split():
        res1 = lemmatizer.lemmatize(word,pos='a')
        res2 = lemmatizer.lemmatize(res1,pos='r')
        res3 = lemmatizer.lemmatize(res2,pos='n')
        res4 = lemmatizer.lemmatize(res3,pos='v')
        result = result + " " + res4

    return result

def lemmatize(data):
    print("-"*16)
    print("Lemmatizing text")
    print("-"*16)
    
    data['lemmatized'] = data['decontracted'].apply(lambda text: lemmatization(text))
    
    print("Text lemmatized!")
    
    return data

def split_data(data):
    print("-"*18)
    print("Splitting the data")
    print("-"*18)
    
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    X = data["lemmatized"]
    y = data[labels].to_numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Splitting done!")
    
    return X_train, X_val, y_train, y_val

