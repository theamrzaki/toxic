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
def train():
    start = timeit.default_timer()
    df = pd.read_csv('train/train.csv',delimiter=',',header=None)

    print ("start training ")
 

    X_train_raw  =    df[2]
    y_train      =    df[3]


    X_train_raw_new = []
    y_train_new = []

    j=0
    for a in X_train_raw :
        #if len(a.split(" ")) < 2:
        #    filter=56
        #else:
        X_train_raw_new.append(  X_train_raw[j])
        y_train_new.append(  y_train[j])

        j+=1




    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw_new)
    #classifier = LogisticRegression(solver ='lbfgs')
    #classifier.fit(X_train, y_train_new)

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    #classifier = RandomForestClassifier(n_jobs=2, random_state=0)
    
    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)
    #classifier.fit(X_train, y_train_new)

    classifier = OneVsOneClassifier(LinearSVC())
    classifier.fit(X_train, y_train_new)

    f = open("logistic_complain_vectorizer.pickle", 'wb')
    pickle.dump(vectorizer, f)
    f.close()

    f = open("logistic_complain_classfier.pickle", 'wb')
    pickle.dump(classifier, f)
    f.close()

    stop = timeit.default_timer()

    print ("training time took was : ")
    print stop - start




#-------------------run----------------------------#

def validation():
    #f = open("logistic_vectorizer.pickle", 'rb')
    #vectorizer = pickle.load(f)
    #f.close()

    ## then load tf-idf of dataset
    #f = open("logistic_classfier.pickle", 'rb')
    #classifier = pickle.load(f)
    #f.close()



    f = open("logistic_complain_vectorizer.pickle", 'rb')
    vectorizer = pickle.load(f)
    f.close()

    # then load tf-idf of dataset
    f = open("logistic_complain_classfier.pickle", 'rb')
    classifier = pickle.load(f)
    f.close()


    df = pd.read_csv('validation.csv',header=None)
    final_result = []

    i = 0 
    try:
        
        for index, row in df.iterrows():
            try:

                text = row[2]
                result = row[3]

                X_test = vectorizer.transform( [text] )
                sentiment = classifier.predict(X_test)

                sentiment_answer=""
                complain_answer=""
            
                sentiment_answer = str(sentiment[0]) 

                
                if sentiment_answer == str(result):
                        final_result.append(1)
                else:
                        final_result.append(0)
    

            except:
                aaa=465
    
            i += 1
    except:
       #f = open("data_etislat_classify_new.pickle", 'wb')
       #pickle.dump(data, f)
       #f.close()  
       aa=654

    
    aa=654
    return  final_result


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
test()
print r


