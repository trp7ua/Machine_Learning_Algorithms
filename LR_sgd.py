import collections
import math
import sys

N = 133       # Change this to present the number of training instances.
eta0 = 0.1      # Initial learning rate; change this if desired.

def update(W, X, l, eta):
    # Compute the inner product of features and their weights.
    a = sum([W[x] for x in X])

    # Compute the gradient of the error function (avoiding +Inf overflow).
    g = ((1. / (1. + math.exp(-a))) - l) if -100. < a else (0. - l)

    # Update the feature weights by Stochastic Gradient Descent.
    for x in X:
        W[x] -= eta * g

def train(fi):
    t = 1
    W = collections.defaultdict(float)
    # Loop for instances.
    for line in fi:
        fields = line.split(',')
        update(W, fields[1:], float(fields[0]), eta0 / (1 + t / float(N)))
        t += 1
    return W

def classify(W, X):
    #print W
    return 1 if 0. < sum([W[x] for x in X]) else 0

def test(W, fi):
    m = 0
    n = 0
    for line in fi:
        fields = line.split(',')
        l = classify(W, fields[1:])
        m += (1 - (l ^ int(fields[0])))
        n += 1
    print('Accuracy = %f (%d/%d)' % (m / float(n), m, n))

if __name__ == '__main__':
    # ./logistic_regression_sgd.py test.txt < train.txt
    ftrain = open('./data/binary_train.csv')
    ftest = open('./data/binary_test.csv')
    W = train(ftrain)
    print W
    #if 1 < len(sys.argv):
    test(W, ftest)
    #for name, value in W.iteritems():
    #    print('%f\t%s' % (value, name))