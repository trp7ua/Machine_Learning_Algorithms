from kernels import *

import numpy as np
from cvxopt.solvers import qp
from cvxopt.base import matrix
import random

class SVM(object):
    def __init__(self, kernel_type='linear', C=1.0, with_slack=False,
                 degree=3.0, sigma=0.7, k=1.0, coef0=0.0):
        if kernel_type=='linear':
            self.kernel = linear_kernel()
        elif kernel_type=='polynomial':
            self.kernel = polynomial_kernel(degree)
        elif kernel_type=='rbf':
            self.kernel = radial_kernel(sigma)
        elif kernel_type=='sigmoid':
            self.kernel = sigmoid_kernel(k, coef0)
        else:
            raise ValueError('Kernel '+kernel_type+' not available')

        self.epsilon = 10**-5
        self.C = C
        self.with_slack = with_slack

    def train(self, samples):
        return self._solve_optimization(samples)

    def indicator(self, sample):
        return sum([sv[0]*sv[1][1]*self.kernel(sample,sv[1][0])
                    for sv in self.support_vector])

    def predict(self, sample):
        if self.indicator(sample)>0:
            return 1
        else:
            return -1

    def _build_P(self, samples):
        return [[si[1]*sj[1]*self.kernel(si[0],sj[0]) for sj in samples]
                for si in samples]

    def _solve_optimization(self, samples):
        P = self._build_P(samples)
        q = [-1.0] * len(samples)

        if self.with_slack:
            h = [0.0] * len(samples) + [self.C] * len(samples)
            G = np.concatenate((np.identity(len(samples)) * -1,
                                np.identity(len(samples))))
        else:
            h = [0.0] * len(samples)
            G = np.identity(len(samples)) * -1

        optimized = qp(matrix(P), matrix(q), matrix(G), matrix(h))
        if optimized['status'] == 'optimal':
            alphas = list(optimized['x'])

            self.support_vector = [(alpha, samples[i])
                                   for i,alpha in enumerate(alphas)
                                   if alpha>self.epsilon]
            return True
        else:
            print "No valid separating hyperplane found"
            return False


def loadCsv(filename):
    #lines = csv.reader(open(filename, "rb"))
    #dataset = list(lines)
    
    fi = open(filename)
    dataset = []
    for line in fi:
        l = line.split(',')
        #l = l[1:] + [l[0]]
        #print l
        dataset.append(l)

    for i in range(len(dataset)):
        #print dataset[i]
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

data = loadCsv('./data/final_data.csv')
train, test = splitDataset(data, 0.75)

clf = SVM('polynomial', with_slack=False, degree=2)
clf.train(train)

print clf.predict(test[0])
#y_test = [clf.predict(sample) for sample in test]