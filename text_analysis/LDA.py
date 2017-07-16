from gensim import corpora, models, similarities
from itertools import chain
from collections import Counter
from nltk.corpus import stopwords


f = open("../movie-reviews-sentiment.tsv")
#f = open("merged_data")

stop = stopwords.words('english')


documents = []
user = set()
#documents = set()

even = []
odd = []

count = 0

#f = f.readlines()

for line in f:
	count += 1
	#if (count > 20):
	#	break
	l = line.split('\t')
	if (l[0] in 'negative'):
		documents.append(l[1])
	#documents.add(l[3])

#documents = list(documents)



temp = []


for sentence in documents:
	sentence = sentence.translate(None, "#&,.();\/-:!$?_'\"")
	sentence = [i for i in sentence.split() if (i not in stop )]
	sentence = ' '.join(sentence)
	temp.append(sentence)
	#for i in sentence:
	#	temp.append(i)

	
documents = temp


#a = Counter(documents).most_common(200)
#for i in a:
#	print i[::-1]




stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]


all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]


id2word = corpora.Dictionary(texts)

mm = [id2word.doc2bow(text) for text in texts]

lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=50, \
                               update_every=1, chunksize=10000, passes=1,\
                               alpha='symmetric', eta=None, decay=0.5, eval_every=10, iterations=10, gamma_threshold=0.001)

for top in lda.print_topics():
  print top
print

lda_corpus = lda[mm]


scores = list(chain(*[[score for topic,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print threshold
#print



#cluster1 = [j for i,j in zip(lda_corpus,documents) if i[0][1] > threshold]
#cluster2 = [j for i,j in zip(lda_corpus,documents) if i[1][1] > threshold]
#cluster3 = [j for i,j in zip(lda_corpus,documents) if i[2][1] > threshold]

#print cluster1
#print cluster2
#print cluster3

