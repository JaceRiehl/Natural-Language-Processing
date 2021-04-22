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
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier

document = [(movie_reviews.words(file),category) for file in movie_reviews.fileids() for category in movie_reviews.categories(file)]
# Randomizes document files so that the data doesnt bias.
shuffle(document)
userInput = []
for input in open("classifyUserInput.txt"):
    userInput.append(word_tokenize(input))

wordFreq = FreqDist(movie_reviews.words())
frequent_words_non_filtered = list(wordFreq)
# Use this line instead of the prior to only select the top 5000 words. It's much faster than using all words
# frequent_words_non_filtered = list(wordFreq)[:5000]
frequent_words_list = [word for word in frequent_words_non_filtered if word not in stopwords.words('english')]

# Finds features from the list of words and places them in a dictionary.
def find_freq_words(word_list):
    words_dict = {}
    for x in frequent_words_list:
        words_dict[x] = x in word_list
    return words_dict


cleaned_documents = [(find_freq_words(set(word_list)),category) for (word_list,category) in document]
training_set = cleaned_documents[:math.floor(len(cleaned_documents) * 0.7)]
testing_set = cleaned_documents[math.floor(len(cleaned_documents) * 0.7):]


classifier = SklearnClassifier(LogisticRegression()).train(training_set)

accurate = 0
actualVals = collections.defaultdict(set)
testVals = collections.defaultdict(set)
i=0
for doc in testing_set:
    guess = classifier.classify(doc[0])
    actual = doc[1]
    actualVals[actual].add(i)
    testVals[guess].add(i)
    print('Guess = ', guess, ' Actual = ', actual)
    if guess == actual:
        accurate = accurate + 1
    i = i+1

print("# of Documents:", len(cleaned_documents))
print('Correct Predictions:', accurate, "/", len(testing_set))
# Using my own accuracy rather than nltks accuracy function because it's much faster
print("Accuracy:", accurate/len(testing_set))
print('Positive Precision:', precision(actualVals['pos'], testVals['pos']))
print('Positive Recall:', recall(actualVals['pos'], testVals['pos']))
print('Positive F-Measure:', f_measure(actualVals['pos'], testVals['pos']))
print('Negative Precision:', precision(actualVals['neg'], testVals['neg']))
print('Negative Recall:', recall(actualVals['neg'], testVals['neg']))
print('Negative F-Measure:', f_measure(actualVals['neg'], testVals['neg']))
print('')
print("Your reviews are:", userInput)

# Classify User input from file.
cleaned_input = [find_freq_words(set(word_list)) for (word_list) in userInput]
i = 0
guess = classifier.classify(cleaned_documents[0][0])
for input in cleaned_input:
    classifierGuess = classifier.classify(input)
    print("User Input:", userInput[i])
    print("is classified as:", classifierGuess)
    i = i+1
