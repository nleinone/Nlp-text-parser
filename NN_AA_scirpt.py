import nltk

import os
import sys
import time
import datetime
import matplotlib.pyplot as plt

import multiprocessing as mp
import sklearn
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import importlib
#Preprocess learning and test data for naivebayes algorithm
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

def co_occ_combiner(lista1):
    co_occuring_list = []
    for i in range(len(lista1)):

        word_set = lista1[i]
        word = word_set[0]
        freq = word_set[1]

        for t in range(len(lista1) - i):
            y = t
            if(i > len(lista1)):
                break
            temp_list = lista1[i + 1:]
            try:
                other_word_set = temp_list[t]
            except Exception:
                break
            other_word = other_word_set[0]
            other_freq = other_word_set[1]
            if(other_word == word):
                co_occuring_list.append([word,freq + other_freq])
    print("\nCo-Occuring top words: \n")
    co_occuring_list.sort()
    for i in range(len(co_occuring_list)):
        print(str(co_occuring_list[i][0]) + ";" + str(co_occuring_list[i][1]) + "\n")

def word_analyser():

    stopwords_file = 'stopwords.txt'
    
    custom_stopwords = set(open(stopwords_file, 'r', encoding = "utf8").read().splitlines())
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    all_stopwords = default_stopwords | custom_stopwords


    total_word_list = []
    #5 top frequent positive words from positive test set used in naivebayes
    print("Collecting info from filtered_test_pos.txt")
    file = open("filtered_test_pos.txt", "r", encoding = "utf8")
    words = nltk.word_tokenize(file.read())
    words = [word.lower() for word in words]
    words = [word for word in words if len(word) > 1]
    words = [word for word in words if not word.isnumeric()]
    words = [word for word in words if word not in all_stopwords]

    fdist = nltk.FreqDist(words)
    print("Top 5 positive words:")
    for word, frequency in fdist.most_common(10):
        print('{};{}'.format(word, frequency))
        total_word_list.append([word, frequency])
    #5 top frequent negative words from positive test set used in naivebayes
    print("Collecting info from filtered_test_neg.txt")
    file = open("filtered_test_neg.txt", "r", encoding="utf8")
    words = nltk.word_tokenize(file.read())
    words = [word.lower() for word in words]
    words = [word for word in words if len(word) > 1]
    words = [word for word in words if not word.isnumeric()]
    words = [word for word in words if word not in all_stopwords]

    fdist = nltk.FreqDist(words)
    print("\nTop 5 negative words:")

    for word, frequency in fdist.most_common(10):
        print('{};{}'.format(word, frequency))
        total_word_list.append([word, frequency])

    #print top co-occuring words:

    #Combine list of most positive and negative words
    co_occ_combiner(total_word_list)

    #report file template
    #ts = time.time()
    #sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')
    #report_file = open("report.txt", 'a')
    #report_file.write("\nTesting sequence at: " + sttime + "\n5 most frequent positive words:" + "\n" + best_positive + "\n5 words with highest coefficient:")

def naiveBayesSentiment(training_data, test_data, data_size):

    #Clean the data
    #print("train_data %d" % len(training_data))
    #print("test_data %d" % len(test_data))
    train_data_clean = preprocess_reviews(training_data)
    test_data_clean = preprocess_reviews(test_data)
    #print(len(train_data_clean))

    #for string in train_data_clean[0:30]:
    #    print(string)

    #for string in train_data_clean[0:30]:
    #    print(string)
    #Vectorization
    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    cv = CountVectorizer(binary=True)
    cv.fit(train_data_clean)

    X = cv.transform(train_data_clean)
    X_test = cv.transform(test_data_clean)

    #Modeling
    #print(data_size)
    print("Performing test split...")
    target = [1 if i < (data_size) else 0 for i in range(data_size*2)]

    print(X.shape)
    print(len(target))

    X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.80)

    for c in [0.01, 0.05, 0.25, 0.5, 1]:

        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

    #Creating the final model with c = 0.05

    print("Creating final model...")
    final_model = LogisticRegression(C=1)
    final_model.fit(X, target)

    predictions = final_model.predict(X_test)
    print(len(predictions))
    accuracy = accuracy_score(target, predictions)

    print ("\nFinal Accuracy: " + str(accuracy) )
    # Final Accuracy:

    feature_to_coef = {word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}
    coeff_total = 0
    counter = 0
    print("\nTotal Coefficient: ")
    for best_positive in sorted(feature_to_coef.items()):
        coeff_total = coeff_total + best_positive[1]
        counter = counter + 1

    #mean of coefficient values, while > 0: more positive sentiment, < 0: more negative sentiment
    print((coeff_total / counter))

    #Collect words with coefficient values in a file.

    print("\n5 words with highest coefficient:")
    for best_positive in sorted(
        feature_to_coef.items(),
        key=lambda x: x[1],
        reverse=True)[:10]:
        print (best_positive)

    print("\n5 words with lowest coefficient:")
    for best_positive in sorted(
        feature_to_coef.items(),
        key=lambda x: x[1])[:10]:
        print (best_positive)
        #report_file = open("report.txt", 'a')
        #report_file.write("\n5 words with lowest coefficient:" + "\n" + best_positive + "\n5 words with highest coefficient:")
        #report_file.write("End of report.")

def create_filtered_file(dir, data_size, file_name, stop_words):
    training_data = []
    c = 0
    with open(dir, 'r',  encoding="utf8") as myfile:
        for i in range(data_size):
            #read (data size) amount of lines from the file
            line = myfile.readline()
            while line:
                words = line.split(" ")
                for r in words:
                    if not r in stop_words:
                        appendFile = open(file_name,'a', encoding="utf8")
                        appendFile.write(" "+r)
                        if(r == "__label__2"):
                            appendFile.write(" "+r)
                training_data.append(line.split())
                appendFile.close()
                line = myfile.readline()
                if(c == data_size):
                    break
                c = c + 1
        myfile.close()
    return training_data

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

def tokenization(file):

    file = open(file, "r", encoding = "utf8")

    stopwords_file = 'stopwords.txt'
    
    custom_stopwords = set(open(stopwords_file, 'r', encoding = "utf8").read().splitlines())
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    all_stopwords = default_stopwords | custom_stopwords


    total_word_list = []

    words = nltk.word_tokenize(file.read())
    words = [word.lower() for word in words]
    words = [word for word in words if len(word) > 1]
    words = [word for word in words if not word.isnumeric()]
    words = [word for word in words if word not in all_stopwords]
    tagged = nltk.pos_tag(words)

    NN_list = list()
    JJ_list = list()

    for tag in tagged:

        if(tag[1] == 'NN'):

            NN_list.append(tag[0])

        if(tag[1] == 'JJ'):

            JJ_list.append(tag[0])

    return NN_list, JJ_list

def mostFrequent(liste, number):

    dist = FreqDist(liste)
    
    common = dist.most_common(number)

    frequent_list = list()

    for i in common:

        frequent_list.append(i)

    return frequent_list

def sentiment(liste):

    sia = SentimentIntensityAnalyzer()

    positive_list = list()
    negative_list = list()

    for i in liste:

        pol_score = sia.polarity_scores(i[0])

        if pol_score["pos"] == 1.0:

            positive_list.append(i[0])

        if pol_score["neg"] == 1.0:

            negative_list.append(i[0])

    return positive_list, negative_list

def plots(one_list, title, xlabel, ylabel):

    words = []
    numbers = []

    for i in one_list:

        words.append(i[0])
        numbers.append(i[1])

    plt.bar(words, numbers)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == '__main__':

    #Find file directory address from data folder
    print("Fetching data...")
    dir = "./amazonreviews"
    for file in os.listdir(dir):
        if file.endswith(".txt"):
            file_dir = os.path.join(dir, file)

    #size of test and train data (in lines) 60 = 30 positive and 30 negative reviews from the positive/negative review data files
    data_size = 25020

    #Training Data: Total of 3600 000 reviews can be used, 1800 000 positive reviews (ordered_train_pos.txt)
    # and 1800 000 negative reviews (ordered_train_neg.txt).
    #If datasize is 60, 30 positive reviews and 30 negative reviews are used.
    training_data = []

    #Test data: Total of 400 000 reviews, 60 = 30 positive and 30 negative reviews from the positive/negative review data files
    #200 000 positive reviews (ordered_test_pos.txt) and 200 000 negative reviews (ordered_test_neg.txt)
    test_data = []

    #Using standard english stopword filter
    stop_words = set(stopwords.words('english'))

    print("Creating test data size of %d reviews..." % data_size)

    #training_data.append(create_filtered_file("./amazonreviews/train/ordered_train_pos.txt", data_size, "filtered_train_pos.txt", stop_words))
    print("filtered_train_pos.txt created. [1/4]")
    #training_data.append(create_filtered_file("./amazonreviews/train/ordered_train_neg.txt", data_size, "filtered_train_neg.txt", stop_words))
    print("filtered_train_neg.txt created. [2/4]")
    #test_data.append(create_filtered_file("./amazonreviews/test/ordered_test_pos.txt", data_size, "filtered_test_pos.txt", stop_words))
    print("filtered_test_pos.txt created. [3/4]")
    #test_data.append(create_filtered_file("./amazonreviews/test/ordered_test_neg.txt", data_size, "filtered_test_neg.txt", stop_words))
    print("filtered_test_neg.txt created. [4/4]")

    #Test Data
    #print("Starting word analyser...")
    #Calculates the words collected in the positive and negative test sets
    #word_analyser()

    print("Beginning")

    NN_list, JJ_list = tokenization("filtered_train_neg.txt")

    #print("NN_list", NN_list)
    #print("JJ_list", JJ_list)
    
    NN_freq = mostFrequent(NN_list, 20)

    plots(NN_freq, "20 Most frequent nouns in negative reviews", "words", "frequency")

    #plt.plot(NN_freq)
    #plt.show()

    print("NN_freq", NN_freq)
    
    JJ_freq = mostFrequent(JJ_list, 20)

    plots(JJ_freq, "20 Most frequent adjectives in negative reviews", "words", "frequency")

    print("JJ_freq", JJ_freq)
    
    NN_sent = sentiment(NN_freq)
    JJ_sent = sentiment(JJ_freq)

    print("NN_sent", NN_sent, "JJ_sent", JJ_sent)



    
    #print("\nProcessing NaiveBayes Algorithm to analyse sentiment...")
    #training_data.extend(create_data_test("filtered_train_pos.txt"))
    #training_data.extend(create_data_test("filtered_train_neg.txt"))
    #test_data.extend(create_data_test("filtered_test_pos.txt"))
    #test_data.extend(create_data_test("filtered_test_neg.txt"))
    #naiveBayesSentiment(training_data, test_data, data_size)

    #print("training data len: " + str(len(training_data)))
    #print("testing data len: " + str(len(test_data)))
    #print("actual data len: " + str(len(actual_data)))

#REFERENCES:
#https://pymotw.com/2/multiprocessing/basics.html
#https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
#https://opensourceforu.com/2016/12/analysing-sentiments-nltk/
#https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
#https://www.nltk.org/_modules/nltk/sentiment/vader.html SOME AUTHORS TO CITE

#CLASSIFIER INFO:
#https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44
