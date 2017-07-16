from __future__ import division
from collections import defaultdict
import math


#Calculate Accuracy
def getErrorPercentage(predicted,true):
    count=0
    for i,j in zip(true,predicted):
        if i==j: count+=1
        #confusion[i,j] += 1
    return (float)(count)/len(true)#,confusion


#Calculate precision and recall 
def getPrecisionandRecall(predictedLabel,trueLabels):
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	# precision = defaultdict(float)
	# recall = defaultdict(float)
	# fmeasure = defaultdict(float)
	uniqLabels = set(trueLabels)

	if len(uniqLabels) > 2:
		print 'Precision and Recall are only defined for binary classifiers'

	for predicted,actual in zip(predictedLabel,trueLabels):
		if actual == 1:
			if predicted == actual:
				tp += 1
			else:
				fp += 1
		if actual == 0:
			if predicted != actual:
				fn += 1
			else:
				tn += 1


	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	fmeasure = 2*precision*recall/(precision + recall)

	# precision[0] = tn/(tn+fn)
	# recall[0] = tn/(tn+fp)
	# fmeasure[0] = 2*precision[0]*recall[0]/(precision[0] + recall[0])

	return (precision,recall,fmeasure)

def rms_error(predictions, targets):
    return math.sqrt(ms_error(predictions, targets))

def ms_error(predictions, targets):
    return mean([(p - t)**2 for p, t in zip(predictions, targets)])

def mean_error(predictions, targets):
    return mean([abs(p - t) for p, t in zip(predictions, targets)])

def mean_boolean_error(predictions, targets):
    return mean([(p != t)   for p, t in zip(predictions, targets)])

def cross_validation(learner, dataset, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; If trials>1, average over several shuffles."""
    if k == None:
        k = len(dataset.examples)
    if trials > 1:
        return mean([cross_validation(learner, dataset, k, trials=1)
                     for t in range(trials)])
    else:
        n = len(dataset.examples)
        random.shuffle(dataset.examples)
        return mean([train_and_test(learner, dataset, i*(n/k), (i+1)*(n/k))
                     for i in range(k)])
    
def leave1out(learner, dataset):
    "Leave one out cross-validation over the dataset."
    return cross_validation(learner, dataset, k=len(dataset.examples))

def learningcurve(learner, dataset, trials=10, sizes=None):
    if sizes == None:
        sizes = range(2, len(dataset.examples)-10, 2)
    def score(learner, size):
        random.shuffle(dataset.examples)
        return train_and_test(learner, dataset, 0, size)
    return [(size, mean([score(learner, size) for t in range(trials)]))
            for size in sizes]




