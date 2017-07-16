from collections import defaultdict,Counter
import numpy as np
import math

#a = [[1,2,3],[4,5,6]]
#a = np.array(a)
f = open("./data/binary_data.csv")
a = []
x = []
y = []
count = 0
for line in f:
    count += 1
    if count > 10:
        break
    temp = [int(float(i)) for i in line.split(',')]
    a.append(temp)
    x.append(temp[1:])
    y.append(temp[0])
#a = np.array(a)
x = np.array(x)
y = np.array(y)


def information_gain(x, y):

    def _entropy(values):
        counts = np.bincount(values)
        probs = counts[np.nonzero(counts)] / float(len(values))
        return - np.sum(probs * np.log(probs))

    def _information_gain(feature, y):
        print np.nonzero(feature)
        feature_set_indices = np.nonzero(feature)[0]
        feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices]
        entropy_x_set = _entropy(y[feature_set_indices])
        entropy_x_not_set = _entropy(y[feature_not_set_indices])
        return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
                                 + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))

    feature_size = x.shape[0]
    feature_range = range(0, feature_size)
    entropy_before = _entropy(y)
    information_gain_scores = []

    for feature in x.T:
        information_gain_scores.append(_information_gain(feature, y))
    return information_gain_scores, []

print information_gain(x,y)
