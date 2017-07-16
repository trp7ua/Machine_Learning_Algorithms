import numpy as np
import random
import pickle

dataset = [[1,20,1], [2,21,0], [3,22,1]]

# it zips column by column
print zip(*dataset)
# [(1, 2, 3), (20, 21, 22), (1, 0, 1)]

'''
pickle
with open('./data/sentences.pickle', 'wb') as handle:
	pickle.dump(X, handle)
with open('./data/sentences.pickle', 'rb') as handle:
	X = pickle.load(handle)
'''


def read_data(path, ignore_header=True,  max_line=-1):
    """ Reads data from file. """
    csv_file_object = csv.reader(open(path, 'rb'))
    if ignore_header:
        header = csv_file_object.next()
    x = []
    for row in csv_file_object:
        if max_line >= 0 and len(row) >= max_line:
            break
        x.append(row)
    return x

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def get_mean_std_feature(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	return summaries

#uniform
print random.uniform(-1,1)
mu = 0
sigma = 2
print np.random.normal(mu, sigma)


#np.genfromtxt('myfile.csv',delimiter=',')

np.vstack(([1,2,3],[4,5,6]))
#array([[1, 2, 3],
#       [4, 5, 6]])
np.column_stack(([1,2,3],[4,5,6]))
#array([[1, 4],
#       [2, 5],
#       [3, 6]])
np.hstack(([1,2,3],[4,5,6]))
#array([1, 2, 3, 4, 5, 6])

# shuffle lines of a file
# cat yourfile.txt | while IFS= read -r f; do printf "%05d %s\n" "$RANDOM" "$f"; done | sort -n | cut -c7- >> new_file
# wc -l < file_name for # of lines
# head -n K file_name > out, tail -n K file_name > out for splitting files with K number of lines top to bottom
# codecs.open(trainfile,'r','utf8')

def rms_error(predictions, targets):
    return math.sqrt(ms_error(predictions, targets))

def ms_error(predictions, targets):
    return mean([(p - t)**2 for p, t in zip(predictions, targets)])

def mean_error(predictions, targets):
    return mean([abs(p - t) for p, t in zip(predictions, targets)])

def mean_boolean_error(predictions, targets):
    return mean([(p != t)   for p, t in zip(predictions, targets)])

def categorizeX(X,Y):
	labels = list(np.unique(Y))
	labels_map = {}
	count = 0
	for i in labels:
		try:
			int(i)
		except ValueError:
			print "labels not int/num"
			labels_map[i] = count
			count += 1


############################## HANDLING MISSING VALS #################################################

# http://students.mimuw.edu.pl/~pbechler/numpy_doc/user/basics.io.genfromtxt.html
# http://nbviewer.ipython.org/github/rasbt/python_reference/blob/master/tutorials/numpy_nan_quickguide.ipynb#Sections
# missing values in genfromtext if are there then are repaced with nan vales to test 
# whether missing values are there check np.isnan(val)
# create a mask if say data's name is data: then the mask for each element: np.isnan(data): it is a bool table 
# use np.count_nonzero(np.isnan(data)) 
# with table with nans: np.sum does not work so use: np.nansum(data)
# column sum: np.nansum(data, axis=0) and axis=1 for row
# remove all those rows which has missing values: ary[~np.isnan(data).any(1)]
# convert missing vals to 0: data0 = np.nan_to_num(data)

def handle_missing_data():
	data=np.genfromtxt('./data/missing_val_data.csv',delimiter=',',dtype=float,invalid_raise=False,
               missing_values='',
               usemask=False,
               filling_values=9999999999999)[:,:-1]

	for i in data:
		for k in i:
			if (k==9999999999999):
				print "yo"

handle_missing_data()