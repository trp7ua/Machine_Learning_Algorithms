from gensim import corpora, models, similarities
from nltk.corpus import stopwords



f= open("only_tweets")
documents = []
stop = stopwords.words('english')
unique = set()

for line in f:
	l = line.strip()
	if not l in unique:
		unique.add(l)
		documents.append(l)


temp = []


for sentence in documents:
	sentence = [i for i in sentence.split() if i not in stop]
	sentence = ' '.join(sentence)
	temp.append(sentence)

	
documents = temp


"""
documents = []
stop = stopwords.words('english')

for line in f:
	documents.append([i for i in line.split() if i not in stop])
"""



stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]

id2word = corpora.Dictionary(texts)
mm = [id2word.doc2bow(text) for text in texts]


lsi = models.LsiModel(corpus=mm, id2word=id2word, num_topics=40)

print lsi.print_topics(30)

"""
for top in lsi.print_topics():
  print top
print

lsi_corpus = lsi[mm]


scores = list(chain(*[[score for topic,score in topic] \
                      for topic in [doc for doc in lsi_corpus]]))
threshold = sum(scores)/len(scores)
print threshold
print

"""
"""
datasets = [[word for word in text] for text in documents]
dictionary = corpora.Dictionary(datasets)
corp = [dictionary.doc2bow(text) for text in datasets]

lsi = models.LsiModel(corp, num_topics=20)

print lsi.show_topics()

"""