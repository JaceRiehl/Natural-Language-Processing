from nltk.corpus import movie_reviews
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.util import pad_sequence
import nltk
from nltk import FreqDist
import math
from random import shuffle
from nltk.metrics import precision, recall, f_measure
from nltk.corpus import stopwords
import collections
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split

# For the classifier, to tell between pos and neg since the NN uses numbers.
posNegDict = {'pos': 0, 'neg': 1}
numToCatDict = {1: 'pos', 0: 'neg'}

# collapse and average the word2vecs keyvaluepair and remove words that arent in the vocab.
def averageVectors(vec, words):
    words = [wd for wd in words if wd in vec.wv.index_to_key]
    if len(words) != 0:
        return np.average(vec.wv[words], axis=0)
    else:
        return None

# Gather the documents with their classifications in numeric form.
document = [(Word2Vec(movie_reviews.sents(file), min_count=1), movie_reviews.words(file), posNegDict[category]) for file in movie_reviews.fileids() for category in movie_reviews.categories(file)]
# Randomizes document files so that the data doesnt bias.
shuffle(document)

# Gather user input
userInput = []
userRaw = []
i = 0
for input in open("classifyUserInput.txt"):
    userRaw.append(word_tokenize(input))
    userInput.append(averageVectors(Word2Vec(word_tokenize(input)), userRaw[i]))
    i = i + 1

# Separate the vectors and the classificiation
x = np.array([averageVectors(x[0], x[1]) for x in document])
y = np.array([y[2] for y in document])

# Prepare neural network and predict.
trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.10)
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5,5,5,5,5,5,5,5,5)).fit(trainX, trainY)
predictions = classifier.predict(testX)


accurate = 0
actualVals = collections.defaultdict(set)
testVals = collections.defaultdict(set)
for i in range(len(predictions)):
    guess = predictions[i]
    actual = testY[i]
    actualVals[numToCatDict[actual]].add(i)
    testVals[numToCatDict[guess]].add(i)
    print('Guess = ', guess, ' Actual = ', actual)
    if guess == actual:
        accurate = accurate + 1

print("# of Documents:", len(predictions))
print('Correct Predictions:', accurate, "/", len(predictions))
print('Positive Precision:', precision(actualVals['pos'], testVals['pos']))
print('Positive Recall:', recall(actualVals['pos'], testVals['pos']))
print('Positive F-Measure:', f_measure(actualVals['pos'], testVals['pos']))
print('Negative Precision:', precision(actualVals['neg'], testVals['neg']))
print('Negative Recall:', recall(actualVals['neg'], testVals['neg']))
print('Negative F-Measure:', f_measure(actualVals['neg'], testVals['neg']))



print("Your reviews are:", userRaw)
userPredictions = classifier.predict(userInput)


for i in range(len(userPredictions)):
    guess = userPredictions[i]
    print("User Input:", userRaw[i])
    if(guess == 0):
        print("is classified as:", "Negative")
    elif(guess == 1):
        print("is classified as:", "Positive")
