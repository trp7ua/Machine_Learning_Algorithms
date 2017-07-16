from Make import makeData, makeDict
from NER_extract import NERCount
from Read import *
from Classification import *
import csv, sys
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.util import ngrams
from time import time
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD as TruncSVD
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.neural_network import BernoulliRBM
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import metrics
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from word2vec_movie import returnWord2VecScores
import pickle
from scipy.sparse import issparse
# Stemming, stopwords, count(words) in a document, unigramns, bigrams,  POS,

initial = time()

stop = stopwords.words('english')
stemmer = PorterStemmer()

"""
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
	if (count > 100):
		break

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

print (a== pos_word2vec_scores), (b == neg_word2vec_scores)

with open('./data/labels.pickle', 'wb') as handle:
	pickle.dump(Y, handle)

with open('./data/sentimentAndOtherFeatures.pickle', 'wb') as handle:
	pickle.dump(sentiment_prob_list, handle)
"""

with open('./data/sentences.pickle', 'rb') as handle:
	X = pickle.load(handle)

with open('./data/words2Vec_feature.pickle', 'rb') as handle:
	(pos_word2vec_scores, neg_word2vec_scores) = pickle.load(handle)


with open('./data/labels.pickle', 'rb') as handle:
	Y  = pickle.load(handle)

with open('./data/sentimentAndOtherFeatures.pickle', 'rb') as handle:
	sentiment_prob_list = pickle.load(handle)


diff = [pos_word2vec_scores_i - neg_word2vec_scores_i for pos_word2vec_scores_i, neg_word2vec_scores_i in zip(pos_word2vec_scores, neg_word2vec_scores)]
max_num = 500
pos_min = -min(pos_word2vec_scores)
neg_min = -min(neg_word2vec_scores)
#print max(neg_word2vec_scores), min(neg_word2vec_scores)
#print max(diff), min(diff)
for i in range(len(pos_word2vec_scores)):
	pos_word2vec_scores[i] += pos_min
	neg_word2vec_scores[i] += neg_min
#a_min = -min(diff)
#diff = [i + a_min for i in diff]




print X[0]


#pos_word2vec_scores, neg_word2vec_scores = np.array(pos_word2vec_scores), np.array(neg_word2vec_scores)
#diff = np.array(diff)
sentiment_prob_list = np.array(sentiment_prob_list)
X = np.array(X)
Y = np.array(Y)

print X.shape
print Y.shape
vectorizer = CountVectorizer(stop_words = stop)
transformer = TfidfTransformer()
#tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  \
#	analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)


trainVectorizerArray = vectorizer.fit_transform(X).toarray()
transformer.fit(trainVectorizerArray)
L = transformer.transform(trainVectorizerArray).toarray()
print L.shape

#tfv.fit(X)
#Z = tfv.transform(X)

#svd = TruncSVD(n_components = 25)
#totalsvd = svd.fit(Z)
#totalsvd = svd.fit_transform(Z)
#totalsvd = totalsvd[:,:] + 1
#X = np.column_stack((L,totalsvd))
#print totalsvd
#print X.shape, Y.shape
X = np.column_stack((L, sentiment_prob_list, pos_word2vec_scores, neg_word2vec_scores))
#X = np.column_stack((L, sentiment_prob_list, diff))
#X = np.column_stack((L, sentiment_prob_list))
if np.any((X.data if issparse(X) else X) < 0):
	print "negative or sparse X"
#X = np.column_stack((X, Z))
#X = trainVectorizerArray
#print X
#sentiment_prob_list = np.array(sentiment_prob_list)
#print "L shape: ", L.shape, "sentiment_prob_list shape: ", sentiment_prob_list.shape, "pos_word2vec_scores, neg shape", pos_word2vec_scores.shape, neg_word2vec_scores.shape
#print pos_word2vec_scores, neg_word2vec_scores
print  "final shape"	
print X.shape, Y.shape

#L = np.column_stack((X,Y))
#np.random.shuffle(L)


#print L.shape



print("---------Training-Data-Read complete---------")
print("Time Taken: %s" % (time()-initial))

length = min([len(i) for i in X])
FinalResult = []
initial0 = time()
sss = StratifiedShuffleSplit(Y, 3, test_size=0.20, random_state=0)
for train_index, test_index in sss:

	#TF-IDF Vectorization

	print('------Vectorizing Data (TF-IDF)------')
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]

	print "Counter of X, Y: train: ", Counter(y_train)
	print "Counter of X, Y: test: ", Counter(y_test)
	#Including small samples category forcefully in training data SET after StratifiedShuffleSplit

	#X_train = np.concatenate((X_train, X_small), axis=0)
	#y_train = np.concatenate((y_train, Y_small), axis=0)
	"""This is imp"""
	#X_train = vectorizer.fit_transform(X_train)
	#X_test = vectorizer.transform(X_test)
	#X_to_predict = vectorizer.transform(X_to_predict1)
	print('------Vectorizing is done------')


	#Feature Selection

	print('------Selecting Features--------')
	"""This is imp"""
	#Selector = SelectKBest(chi2, k=int(length*0.9))
	#print X_train.shape, y_train.shape, X_test.shape, y_test.shape
	#X_train = Selector.fit_transform(X_train, y_train)
	#print np.asarray(vectorizer.get_feature_names())[Selector.get_support()]
	#X_test = Selector.transform(X_test)
	#X_to_predict = Selector.transform(X_to_predict)
	print('---Feature Selection is done---')

	#print('------Selecting Features--------')
	#Selector = BernoulliRBM(n_components=2)
	#X_train = Selector.fit_transform(X_train, y_train)
	#X_test = Selector.transform(X_test)
	#print('---Feature Selection is done---')

	results = []
	
	#clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
	#clf = SVC()

	#clf = Pipeline([('vect', CountVectorizer()),\
	#	('tfidf', TfidfTransformer()),('clf', MultinomialNB()) ])
	clf = MultinomialNB()
	#clf = RandomForestClassifier()

	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)
	

	
	print "accuracy_score", accuracy_score(y_test, y_pred)

	print "roc curve", roc_auc_score(y_test,y_pred)

	print (precision_recall_fscore_support(y_test, y_pred, average=None))

	results.append([0])




