import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB


def read_input(path):
    f = open(path, "r",  encoding='utf-8')
    N= f.readline()
    questions= []
    excerpts = []
    
    for i in range(0,int(N)):
        d= f.readline()
        loaded_json = json.loads(d)
        excerpts.append(loaded_json["excerpt"])
        questions.append(loaded_json["question"]+ loaded_json["excerpt"])

    return (questions,excerpts)

def read_data(path):
    f = open(path, "r",  encoding='utf-8')
    N= f.readline()
    questions= []
    excerpts = []
    label = []
    
    for i in range(0,int(N)):
        d= f.readline()
        loaded_json = json.loads(d)
        label.append(loaded_json["topic"])
        excerpts.append(loaded_json["excerpt"])
        questions.append(loaded_json["question"]+ loaded_json["excerpt"])
    return (questions, excerpts, label)

def simpletest(path):
    f = open(path, "r",  encoding='utf-8')
    label = []
    while(True):
        d= f.readline()
        if(d==""):
            break
        else:
            label.append(d)
       
    return label

if __name__ == '__main__':  
    
    (questions, excerpts, label)=read_data(u"trainin.txt")
    (questions_test, excerpts_test)= read_input(u"input00.txt")

    output0= simpletest(u"Output00.txt")
    
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vect.fit_transform(questions)
    #print(questions)
    x_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #print(x_train_tfidf)
    #print(x_train_tfidf.shape)

    x_test_counts = count_vect.transform(questions_test)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    clf = MultinomialNB().fit(x_train_tfidf, label)
    y_pred=clf.predict(x_test_tfidf)
    #acc = accuracy_score(np.array(output0),y_pred)
    for i in y_pred:
        print(i)


# using pipeline
'''
if __name__ == '__main__':  
    
    (questions, excerpts, label)=read_data(u"training.json")
    (questions_test, excerpts_test)= read_input(u"input00.txt")


    sfc= make_pipeline(CountVectorizer(), TfidfTransformer(),  MultinomialNB())

    #x_test_counts = count_vect.transform(questions_test)
    #x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    sfc.fit(questions, label)
    y_pred=sfc.predict(questions_test)
    #acc = accuracy_score(np.array(output0),y_pred)
    for i in y_pred:
        print(i)
'''

    
  

