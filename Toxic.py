import pandas as pd           #logistic 12:00pm midnight 8/2/2018
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

import pickle
import timeit  # to measure time of training

#--------------train------------------------------------#

train_count      = 1000
validation_count = 100


def train():
    start = timeit.default_timer()
    df = pd.read_csv('train.csv',delimiter=',',header=None)

    print ("start training ")
 
    train, test = train_test_split(df, test_size=0.2)

    X_train_raw          =    train[1]

    y_toxic               =    train[2]
    #y_severe_toxic       =    train[3]
    #y_obscene            =    train[4]
    #y_threat             =    train[5]
    #y_insult             =    train[6]
    #y_identity_hate      =    train[7]


    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)

    #classifier_toxic = LogisticRegression(solver ='lbfgs')
    #classifier_toxic.fit(X_train, y_toxic)

    classifier_toxic = OneVsOneClassifier(LinearSVC())
    classifier_toxic.fit(X_train, y_toxic)

    f = open("logistic_complain_vectorizer.pickle", 'wb')
    pickle.dump(vectorizer, f)
    f.close()


    f = open("logistic_complain_classfier.pickle", 'wb')
    pickle.dump(classifier_toxic, f)
    f.close()


    stop = timeit.default_timer()

    print ("training time took was : ")
    print stop - start


#-------------------run----------------------------#

def validation():

    f = open("logistic_complain_vectorizer.pickle", 'rb')
    vectorizer = pickle.load(f)
    f.close()

    # then load tf-idf of dataset
    f = open("logistic_complain_classfier.pickle", 'rb')
    classifier = pickle.load(f)
    f.close()


    df = pd.read_csv('train.csv',header=None)
    final_result = []
    train, test = train_test_split(df, test_size=0.2)

    i = 0 
    try:
        
        for index, row in test[:10000].iterrows():
            try:

                text    =    row[1]
                result  =    row[2]

                X_test = vectorizer.transform( [text] )
                sentiment = classifier.predict(X_test)

                sentiment_answer = str(sentiment[0]) 

                if sentiment_answer == str(result):
                        final_result.append(1)
                else:
                        final_result.append(0)
    
            except:
                aaa=465
    
            i += 1
    except:
       aa=654

    aa=654
    return  float(final_result.count(1)) / float(10000)


#-------------------run----------------------------#

def test():
  
    f = open("logistic_complain_vectorizer.pickle", 'rb')
    vectorizer = pickle.load(f)
    f.close()

    # then load tf-idf of dataset
    f = open("logistic_complain_classfier.pickle", 'rb')
    classifier = pickle.load(f)
    f.close()


    df = pd.read_csv('real/test.csv')
    #final_result = []
    data= []


    i = 0 
    try:
        
        for index, row in df.iterrows():
            try:

                text = row[2]

                X_test = vectorizer.transform( [text] )
                sentiment = classifier.predict(X_test)

                sentiment_answer=""
                complain_answer=""
            
                sentiment_answer = str(sentiment[0]) 

                
                #if sentiment_answer == str(result):
                #        final_result.append(1)
                #else:
                #        final_result.append(0)
    

                data.append( {'PhraseId': str(row[0]) , 'Sentiment': str(sentiment_answer)})

            except:
                aaa=465
    
            i += 1
    except:
       #f = open("data_etislat_classify_new.pickle", 'wb')
       #pickle.dump(data, f)
       #f.close()  
       aa=654

    
    aa=654

    df = pd.DataFrame(data)
    filename = 'Result/submitOvR.csv'
    df.to_csv(filename, sep=',', encoding='utf-8')

    return  ""

print "training"
train()
print "training done"
r=validation()
print r


