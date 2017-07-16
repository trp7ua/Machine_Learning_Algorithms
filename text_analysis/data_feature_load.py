from NER_extract import NERCount
from Read import *
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.util import ngrams
from random import shuffle
from word2vec_movie import returnWord2VecScores
import pickle
import numpy as np

# Stemming, stopwords, count(words) in a document, unigramns, bigrams,  POS,

stop = stopwords.words('english')
stemmer = PorterStemmer()

def stemmingStop(text):
	#text = ''.join(c for c in text if ord(c)<128)
	#text = ' '.join(stemmer.stem(c.lower()) for c in text.split() if c.isalnum() and c.lower() not in stop and 'traffic' not in c.lower())
	text = ' '.join(stemmer.stem(c.lower()) for c in text.split() if c.lower() not in stop )
	#text = ' '.join(c.lower() for c in text.split() if c.isalnum() and c.lower() not in stop)
	text= nltk.word_tokenize(text)
	text = pos_tag(text)
	p1 = []
	p2 = ['']
	for i in text:
		p1.append(i[0])
		p2.append(i[1])
		#l = ' '.join(i)
		#p.append(l)
	#p1 = ' '.join(stemmer.stem(c) for c in p1)
	#p1 = ' '.join(p1)
	#p1 = nltk.word_tokenize(p1)
	#bigrams = ngrams(p1, 2)
	#print bigrams
	#p1 = []
	#for i in bigrams:
	#	p1.append(' '.join(i))
	#result = []
	#for word in p1:
	#	ss = nltk.wordnet.wordnet.synsets(word)
	#	result.extend(str(s) for s in ss if ".n." not in str(s))
	#p1= result
	#print p1
	text = ' '.join(p1) + ' '.join(p2)
	#text = p1 + ' '.join(p2)
	#text1 = ' '.join(p)
	#print text
	return text

def extractor(text):
    '''Extract the last letter of a word as the only feature.'''
    feats = {}
    last_letter = word[-1]
    feats["last_letter({0})".format(last_letter)] = True
    return feats

def readSentimentList(file_name):
    ifile = open(file_name, 'r')
    happy_log_probs = {}
    sad_log_probs = {}
    ifile.readline() #Ignore title row
    
    for line in ifile:
        tokens = line[:-1].split(',')
        happy_log_probs[tokens[0]] = float(tokens[1])
        sad_log_probs[tokens[0]] = float(tokens[2])

    return happy_log_probs, sad_log_probs

def classifySentiment(words, happy_log_probs, sad_log_probs):
    # Get the log-probability of each word under each sentiment
    happy_probs = [happy_log_probs[word] for word in words if word in happy_log_probs]
    sad_probs = [sad_log_probs[word] for word in words if word in sad_log_probs]

    # Sum all the log-probabilities for each sentiment to get a log-probability for the whole tweet
    tweet_happy_log_prob = np.sum(happy_probs)
    tweet_sad_log_prob = np.sum(sad_probs)

    # Calculate the probability of the tweet belonging to each sentiment
    prob_happy = np.reciprocal(np.exp(tweet_sad_log_prob - tweet_happy_log_prob) + 1)
    prob_sad = 1 - prob_happy

    return prob_happy, prob_sad


f = open("../movie-reviews-sentiment.tsv")

X = []
Y = []
X_Vec = []

sentiment_prob_list = []
happy_log_probs, sad_log_probs = readSentimentList('twitter_sentiment_list.csv')
count = 0

for line in f:
	count +=1
	#if (count > 100):
	#	break

	k = line.split('\t')
	if (k[0] in 'negative'):
		Y.append(0)
	else:
		Y.append(1)
	l = k[1]
	#ner_count = 0
	ner_count = NERCount(l)
	l = l.strip('\n')
	l = stemmingStop(l)
	
	l = l.split()
	happy_prob, sad_prob = classifySentiment(l, happy_log_probs, sad_log_probs)
	
	#print ner_count
	sentiment_prob_list.append([happy_prob, len(l), ner_count])
	X_Vec.append(l)
	l = ' '.join(l) 
	X.append(l)
	#print "yo1"
	#X.append(ner_count)
#print "yo"

pos_word2vec_scores, neg_word2vec_scores = returnWord2VecScores(X_Vec)
del X_Vec

with open('./data/sentences.pickle', 'wb') as handle:
	pickle.dump(X, handle)

with open('./data/words2Vec_feature.pickle', 'wb') as handle:
	pickle.dump((pos_word2vec_scores, neg_word2vec_scores), handle)

with open('./data/labels.pickle', 'wb') as handle:
	pickle.dump(Y, handle)

with open('./data/sentimentAndOtherFeatures.pickle', 'wb') as handle:
	pickle.dump(sentiment_prob_list, handle)