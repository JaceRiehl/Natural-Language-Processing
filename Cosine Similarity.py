from nltk.corpus import movie_reviews
from nltk import word_tokenize
import nltk
from nltk import FreqDist
import math
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def cosSimilarity(d1,d2):
    return np.dot(d1,d2)/(np.linalg.norm(d1) * np.linalg.norm(d2))

document = [(movie_reviews.words(file),category) for file in movie_reviews.fileids() for category in movie_reviews.categories(file)]
posDocuments = [doc for doc in document if doc[1] == "pos"]
negDocuments = [doc for doc in document if doc[1] == "neg"]

posDocs = []
negDocs = []
for i in range(len(posDocuments)):
    posDocs.append(posDocuments[i][0])
    negDocs.append(negDocuments[i][0])

posSentences = [" ".join(list_of_words) for list_of_words in posDocs]
negSentences = [" ".join(list_of_words) for list_of_words in negDocs]


posVectorizer = TfidfVectorizer(stop_words='english')
posTfidf = posVectorizer.fit_transform(posSentences)
posTfidfDocs = posTfidf.toarray()

negVectorizer = TfidfVectorizer(stop_words='english')
negTfidf = negVectorizer.fit_transform(negSentences)
negTfidfDocs = negTfidf.toarray()


posAverage = []
negAverage = []
for i in range(len(posTfidfDocs)):
    for j in range(i+1, len(posTfidfDocs)):
        # print("i = " + str(i) + " j = " + str(j))
        posAverage.append(cosSimilarity(posTfidfDocs[i], posTfidfDocs[j]))
        negAverage.append(cosSimilarity(negTfidfDocs[i], negTfidfDocs[j]))

# print(len(posAverage))
# print(len(negAverage))

def average(ls):
    return sum(ls)/len(ls)


print("Positive Average Similarity: " + str(average(posAverage)))
print("Negative Average Similarity: " + str(average(negAverage)))
