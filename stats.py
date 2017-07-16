import numpy as np
from scipy.stats import kstest
from scipy.stats import ks_2samp
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_1samp
from scipy.stats import chisquare
import scipy 


def corr(X,Y):
	return np.corrcoef(X,Y)[0,1]

def ks2sample(X,Y):
	D,p = ks_2samp(X, Y)
	return (D,p)
# D static 2 sided p value: If the K-S statistic is small or the p-value is high, 
# then we cannot reject the hypothesis that the distributions of the two samples are the same.

def spearman(X,Y):
	rho, p = spearmanr(X,Y)
	return (rho, p)

def tTest(X,m):
	# parametric : particularly considers normal distribution
	t, p = ttest_1samp(x, m)
	return (t,p)

def chi2(X,Y):
	c,p = chisquare(X)
	return (c,p)

def manwhitneyUtest(X,Y):
	# non-parametric T test
	return mannwhitneyu(X,Y)[1]


X = [1,2,3,4]
Y = [4,5,1,3]

print corr(X,Y)
print chi2(X,Y)
D,p = ks2sample(X,Y)
print D,p
print manwhitneyUtest(X,Y)


def rms_error(predictions, targets):
    return math.sqrt(ms_error(predictions, targets))

def ms_error(predictions, targets):
    return mean([(p - t)**2 for p, t in zip(predictions, targets)])

def mean_error(predictions, targets):
    return mean([abs(p - t) for p, t in zip(predictions, targets)])

def mean_boolean_error(predictions, targets):
    return mean([(p != t)   for p, t in zip(predictions, targets)])


def correlationsSorted(X,Y):
	sorted_features = []
	for i in range(X.shape[1]):
		sorted_features.append((corr(X[:,i], Y), i))
	sorted_features = set(list(sorted_features), key=1)
	return sorted_features

f = open('./data/binary_data.csv')
X = []
Y = []
for line in f:
	l = line.split(',')
	X.append(l[1:])
	Y.append(l[0])
X = np.array(X)
Y = np.array(Y)

print correlationsSorted(X,Y)

