
def readData(myfile, threshold):
	from Make import makeDict
	from sklearn.cross_validation import StratifiedShuffleSplit
	import csv, sys
	import numpy as np
	csv.field_size_limit(sys.maxsize)

	X, Y= [], []
	with open(myfile, 'rb') as f:
	    reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    for row in reader:
	    	Y.append(row[0])
	    	X.append(row[1])
	f.close()
	del(reader)

	# Reducing Training data to fit memory
	if myfile=='Traindata_extra.csv':
		unique_Y = makeDict(Y)
		majority_Y = list(True if unique_Y[v]>=threshold else False for i,v in enumerate(Y))
		Y_big = np.array([v for i,v in enumerate(Y) if majority_Y[i]])
		X_big = np.array([v for i,v in enumerate(X) if majority_Y[i]])
		Y_small = np.array([v for i,v in enumerate(Y) if not majority_Y[i]])
		X_small = np.array([v for i,v in enumerate(X) if not majority_Y[i]])
		del majority_Y

	#Y, X  = np.array(Y), np.array(X)

	'''
	if myfile=='Traindata_extra.csv':
		sss = StratifiedShuffleSplit(Y, 1, test_size=0.60, random_state=0)
		for train_index, test_index in sss:
		  X_train, _ = X[train_index], X[test_index]
		  y_train, _ = Y[train_index], Y[test_index]
		X = X_train
		Y = y_train
		del y_train, X_train, sss
	'''
	return (X_big,Y_big, X_small, Y_small)


def readData_to_predict(myfile):
	from Make import makeDict
	import csv, sys
	import numpy as np
	csv.field_size_limit(sys.maxsize)

	X, Y= [], []
	with open(myfile, 'rb') as f:
	    reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    for row in reader:
	    	Y.append(row[0])
	    	X.append(row[2])
	f.close()
	del(reader)

	# Reducing Training data to fit memory


	return (X, Y)

