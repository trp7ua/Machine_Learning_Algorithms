import numpy as np
import random

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


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

def classify(W, X):
    return 1 if 0. < sum([W[x] for x in X]) else 0

def test(W, X, Y):
    m = 0
    n = 0
    for l in X:
        #fields = line.split(',')
        l = classify(W, fields[1:])
        m += (1 - (l ^ int(fields[0])))
        n += 1
    print('Accuracy = %f (%d/%d)' % (m / float(n), m, n))

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
#x, y = genData(100, 25, 10)

dataset = loadCsv('./data/final_data.csv')
train, test = splitDataset(dataset, 0.75)

X_train = []
Y_train = []
for i in train:
    #print i
    X_train.append(i[1:])
    Y_train.append(i[0])

X_test = []
Y_test = []
for i in test:
    X_test.append(i[1:])
    Y_test.append(i[0])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test  = np.array(Y_test)

m, n = np.shape(X_train)
numIterations= 100000
alpha = 0.000000000005
theta = np.ones(n)
theta = gradientDescent(X_train, Y_train, theta, alpha, m, numIterations)
print(theta)

