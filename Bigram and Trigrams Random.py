from nltk.corpus import brown as br
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.util import pad_sequence
import random

# brown.fileids() to get all sentences
tokens = []
for fileid in br.fileids():
    tokens.extend(br.sents(fileid))


sents = []
bigrams = []
startOfSentences = []
trigrams = []
startOfSentencesTrigram = []


# Goes through all of the sentences, creates bi or trigrams and assigns the starting sentences
# to a separate array in order to improve run time, rather than randoming until i finally get a start of sentence.
for tok in tokens:
    tokapp = list(pad_sequence(tok, pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>", n=2))
    big = list(ngrams(tokapp, 2))
    tri = list(ngrams(tokapp, 3))
    for bi in big:
        if bi[0] == "<s>":
            startOfSentences.append(bi)
        else:
            bigrams.append(bi)
    for tr in tri:
        if tr[0] == "<s>":
            startOfSentencesTrigram.append(tr)
        else:
            trigrams.append(tr)
    # sents.append(tokapp)



# Constructing the beginning of sentences.
start = random.randint(0,len(startOfSentences))
startTrigram = random.randint(0,len(startOfSentencesTrigram))
bigramSentence = [startOfSentences[start][0], startOfSentences[start][1]]
trigramSentence = [startOfSentencesTrigram[startTrigram][0], startOfSentencesTrigram[startTrigram][1], startOfSentencesTrigram[startTrigram][2]]


while True:
    ran = random.randint(0,len(bigrams)-1)
    big = bigrams[ran]
    bigramSentence.append(big[0])
    bigramSentence.append(big[1])
    if big[1] == "</s>":
        break

# Trigram
while True:
    ran = random.randint(0,len(trigrams)-1)
    tri = trigrams[ran]
    trigramSentence.append(tri[0])
    trigramSentence.append(tri[1])
    trigramSentence.append(tri[2])
    if tri[2] == "</s>":
        break


# Bigram
# print(bigramSentence)
print("Bigram: ")
for x in bigramSentence:
    print(x, end=" ")
print(" ")
# Trigram

# print(trigramSentence)
print("Trigram: ")
for x in trigramSentence:
    print(x, end=" ")
