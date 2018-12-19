import os
import sys
import time
import datetime

import multiprocessing as mp
import sklearn
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


def split_data(file_dir):
    #Training Data
    training_data = []
    for line in open('./train/train.tf.txt', 'r',  encoding="utf8"):
        training_data.append(line.strip())
    #Test data
    test_data = []
    for line in open('./test/test.tf.txt', 'r',  encoding="utf8"):
        test_data.append(line.strip())
    test_data.append(" ")

    #Actual Data
    actual_data = []
    for line in open(file_dir, 'r', encoding="utf8"):
        actual_data.append(line.strip())
    return [training_data, test_data, actual_data]

def preprocess_reviews(reviews):
    reviews = [line.replace("__label__1", "") for line in reviews]
    reviews = [line.replace("__label__2", "") for line in reviews]
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    reviews = [REPLACE_NO_SPACE.sub("", str(line).lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

def create_data_test(file_dir):
    training_data = []

    with open(file_dir, 'r',  encoding="utf8") as myfile:
        line = myfile.readline()
        while line:
            training_data.append(line)
            line = myfile.readline()
        myfile.close()

    print(len(training_data))
    return training_data

def SGD(training_data,test_data, data_size):
    print("test_data %d" % len(test_data))
    train_data_clean = preprocess_reviews(training_data)
    test_data_clean = preprocess_reviews(test_data)
    print(len(test_data_clean))

    #For stopwords add stop_words=english in parameters
    #tf = TfidfVectorizer(ngram_range=(1,1))# Unigram
    tf = TfidfVectorizer(ngram_range=(2, 2))# Bigram
    #cv=CountVectorizer(binary=True)
    tf.fit(train_data_clean)

    X = tf.transform(train_data_clean)
    X_test = tf.transform(test_data_clean)

    print("Performing test split...")
    target = [1 if i < (data_size) else 0 for i in range(data_size * 2)]

    print(X.shape)
    print(X_test.shape)
    print(len(target))

    X_train, X_val, y_train, y_val = train_test_split(X, target, train_size=0.80)

    #scaler, recommend to scale data
    #source https://scikit-learn.org/stable/modules/sgd.html#classification

    #scaler = StandardScaler()
    #scaler.fit(X_train)  # Don't cheat - fit only on training data
    #X_train = scaler.transform(X_train)
    #X_val = scaler.transform(X_val)



    clf = SGDClassifier().fit(X_train, y_train)

    print(clf.get_params)
    print("accuracy model",accuracy_score(y_val,clf.predict(X_val)))

    f_model=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False).fit(X, target)

    predictions = f_model.predict(X_test)
    accuracy = accuracy_score(target, predictions)
    print("\nFinal Accuracy: " + str(accuracy))
    #print ("score")
    #print("accuracy")

    feature_to_coef = {word: coef for word, coef in zip(tf.get_feature_names(), f_model.coef_[0])}
    coeff_total = 0
    counter = 0
    print("\nTotal Coefficient: ")
    for best_positive in sorted(feature_to_coef.items()):
        coeff_total = coeff_total + best_positive[1]
        counter = counter + 1

    # mean of coefficient values, while > 0: more positive sentiment, < 0: more negative sentiment
    print((coeff_total / counter))

    # Collect words with coefficient values in a file.
    pword=[]
    pcoeff=[]
    nword=[]
    ncoeff=[]
    print("\n5 words with highest coefficient:")
    for best_positive in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1],
            reverse=True)[:10]:
        pword.append(best_positive[0])
        pcoeff.append(best_positive[1])
        print(best_positive)

    print("\n5 words with lowest coefficient:")
    for best_positive in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1])[:10]:
        nword.append(best_positive[0])
        ncoeff.append(best_positive[1])
        print(best_positive)
    plt.bar(pword[:5]+nword[:5],pcoeff[:5]+ncoeff[:5])
    plt.title("highest and lowest coefficients")
    plt.xlabel("word")
    plt.ylabel("coefficient")
    #plt.text(pword[:5]+nword[:5],pcoeff[:5]+ncoeff[:5],s, fontsize=20)
    plt.show()
if __name__ == '__main__':
    data_size = 25020
    training_data = []
    test_data = []

    print("\nProcessing NaiveBayes Algorithm to analyse sentiment...")
    training_data.extend(create_data_test("filtered_train_pos.txt"))
    training_data.extend(create_data_test("filtered_train_neg.txt"))
    test_data.extend(create_data_test("filtered_test_pos.txt"))
    test_data.extend(create_data_test("filtered_test_neg.txt"))
    SGD(training_data, test_data, data_size)