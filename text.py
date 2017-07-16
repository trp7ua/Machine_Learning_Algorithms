from collections import Counter
from itertools import chain
from string import punctuation

from nltk.corpus import brown, stopwords

# Let's say the training/testing data is a list of words and POS
sentences = brown.sents()[:2]

# Extract the content words as features, i.e. columns.
vocabulary = list(chain(*sentences))
stops = stopwords.words('english') + list(punctuation)
vocab_nostop = [i.lower() for i in vocabulary if i not in stops]

# Create a matrix from the sentences
matrix = [Counter([w for w in words if w in vocab_nostop]) for words in sentences]

print matrix

"""
import codecs, re, time
from itertools import chain

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

trainfile = 'train.txt'
testfile = 'test.txt'

# Vectorizing data.
train = []
word_vectorizer = CountVectorizer(analyzer='word')
trainset = word_vectorizer.fit_transform(codecs.open(trainfile,'r','utf8'))
tags = ['bs','pt','es','sr']

# Training NB
mnb = MultinomialNB()
mnb.fit(trainset, tags)

# Tagging the documents
codecs.open(testfile,'r','utf8')
testset = word_vectorizer.transform(codecs.open(testfile,'r','utf8'))
results = mnb.predict(testset)

print results
"""