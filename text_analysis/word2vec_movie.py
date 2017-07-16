import gensim
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support



def returnWord2VecScores(documents):

	f1 = open('posNegWords')

	posneg = []
	for line in f1:
		posneg.append(line)

	"""
	f1 = open('cleanedPosTopics')
	f2 = open('cleanedNegTopics')
	pos = []
	for line in f1:
		pos.append(line)

	neg = []
	for line in f1:
		neg.append(line)

	pos_words = []
	neg_words = []

	for i in pos:
		l = i.split()
		for k in l:
			if k not in pos_words:
				pos_words.append(k)


	for i in neg:
		l = i.split()
		for k in l:
			if l not in neg_words:
				neg_words.append(k)

	"""
	pos_words = posneg[0]
	neg_words = posneg[1]

	#pos_words= pos_words.translate(None, "#'\'&,.\/:-;!$?_")
	#neg_words= neg_words.translate(None, "#'\'&,.\/:-;!$?_")

	pos_words = pos_words.split()
	neg_words = neg_words.split()
	stop = stopwords.words('english')
	
	"""
	f = open("../movie-reviews-sentiment.tsv")
	#f = open("merged_data")

	

	documents = []
	user = set()
	#documents = set()

	even = []
	odd = []

	count = 0

	#f = f.readlines()
	Y = []
	Y_out = []

	for line in f:
		count += 1
		#if (count > 20):
		#	break
		l = line.split('\t')
		if (l[0] in 'negative'):
			Y.append(0)
		else:
			Y.append(1)
		#temp = l[1].translate(None, "#'\'&,.\/:-;!$?_")
		documents.append(l[1])

	print Counter(Y)

	temp = []


	for sentence in documents:
		sentence = [i for i in sentence.split() if (i not in stop)]
		#sentence = ' '.join(sentence)
		temp.append(sentence)
		#for i in sentence:
		#	temp.append(i)	
	documents = temp
	"""
	model = gensim.models.Word2Vec(documents, min_count=1)

	word_embeddings = model.syn0.copy()
	words = model.index2word[:]
	print len(word_embeddings), len(words)

	not_match_key = []
	not_match_key_pos = []
	not_match_key_neg = []

	neg_word2vec_scores = []
	pos_word2vec_scores = []
	#pos_scores = []


	for i in documents:
		sent = i
		pos_score = 0
		neg_score = 0
		sent = ' '.join(sent)
		#sent = sent.translate(None, "#'\'&,.\/:-;!$?_")
		sent = [s for s in sent.split() if (s not in stop)]

		for l in sent:
			for j in pos_words:
				try:
					#j = j.translate(None, "#'\'&,.\/:-;!$?_")
					pos_score += model.similarity(l,j)
				except (KeyError):
					#if (l not in not_match_key):
					#	not_match_key.append(l)
					#if (j not in not_match_key_pos):
					#	not_match_key_pos.append(j)
					pass
			for k in neg_words:
				try:
					#k = k.translate(None, "#'\'&,.\/:-;!$?_")
					neg_score += model.similarity(l,k)
				except (KeyError):
					#if (l not in not_match_key):
					#	not_match_key.append(l)
					#if (j not in not_match_key_neg):
					#	not_match_key_neg.append(k)
					pass

		neg_word2vec_scores.append(neg_score)
		pos_word2vec_scores.append(pos_score)


	return (pos_word2vec_scores, neg_word2vec_scores)
		#if (pos_score >= neg_score):
		#	Y_out.append(1)
		#else:
		#	Y_out.append(0)

#print Counter(Y_out)

#count = 0
#for i in range(len(Y)):
#	if (Y[i] == Y_out[i]):
#		count += 1

#print count

#print (precision_recall_fscore_support(Y, Y_out, average=None))

#print "pos", not_match_key_pos

#print "neg", not_match_key_neg

#print "total", not_match_key

"""

sent = 'first-time writer-director dylan kidd also has a good ear for dialogue , and the characters sound like real people.'
#sent = sent.translate(None, "#&,.\/:!$?_'-;\"")
sent = [i for i in sent.split() if (i not in stop)]
#sent = sent.split()
pos_score = 0
neg_score = 0

for i in sent:
	for j in pos_words:
		try:
			pos_score += model.similarity(i,j)
		except (KeyError):
			pass
	for k in neg_words:
		try:
			neg_score += model.similarity(i,k)
		except (KeyError):
			pass
"""


#print model.similarity(, pos_words)
#print pos_score, neg_score
