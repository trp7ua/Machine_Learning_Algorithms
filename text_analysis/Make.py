import csv,os,string
from nltk.stem.porter import *
from nltk.corpus import stopwords

def makeData():
	stop = stopwords.words('english')
	stemmer = PorterStemmer()
	Trainset_files = []
	Testset_files  = []
	path = 'data/'

	with open('classes.csv', 'rU') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', dialect=csv.excel_tab)
		for row in spamreader:
			file_path = path + row[0] + '.txt'
			if os.path.isfile(file_path):
				f = open(file_path, 'r').read()
				f = ''.join(c for c in f if ord(c)<128)
				f = ' '.join(stemmer.stem(c.lower()) for c in f.split() if c.isalpha() and c.lower() not in stop)
				if row[1]=='To be Tested':
					Testset_files.append([row[0], row[1]] + [f])
				else:
					Trainset_files.append([row[1]] + [f])
	Trainset_files = Trainset_files[1:]

	with open('Traindata_extra.csv', 'wb') as csvfile1:
	    spamwriter1 = csv.writer(csvfile1, delimiter=',', quotechar='|',quoting=csv.QUOTE_MINIMAL)
	    spamwriter1.writerows(Trainset_files)

	with open('Testdata_extra.csv', 'wb') as csvfile2:
	    spamwriter2 = csv.writer(csvfile2, delimiter=',', quotechar='|',quoting=csv.QUOTE_MINIMAL)
	    spamwriter2.writerows(Testset_files)


def makeDict(row):
	mydict ={}
	for i in row:
		if i in mydict:
			mydict[i]+=1
		else:
			mydict[i]=1
	return mydict