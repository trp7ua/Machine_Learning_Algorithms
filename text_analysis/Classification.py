from time import time
from sklearn import metrics

def procedure(clf, X_train, y_train, X_test, y_test, X_to_predict):

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    answer = clf.predict(X_to_predict)

    Fscore = metrics.f1_score(y_test, pred)
    #print("f1-score:   %0.3f" % Fscore)

    #print("confusion matrix:")
    #print(metrics.confusion_matrix(y_test, pred))

    #print("Accuracy classification score:"),
    #print(round(metrics.accuracy_score(y_test, pred),3))

    #print("Recall Rate:"),
    #print(round(metrics.recall_score(y_test, pred),3))

    clf_name = str(clf).split('(')[0]
    return clf_name, Fscore, answer#, train_time, test_time
