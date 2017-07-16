from sklearn import tree
from sklearn import cross_validation
import time
import numpy as np
#from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
from sklearn import linear_model
#from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import preprocessing
#from sklearn.decomposition import RandomizedPCA
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn import gaussian_process
from collections import Counter



X=[]
Y=[]
X_mob=[]
Y_mob=[]
#x1=[]
#1=[]

s = set()


def meanImg(input):
	
	if (input < 165):
		return [1,0,0,0]
	elif (input < 180):
		return [0,1,0,0]
	elif (input < 200):
		return [0,0,1,0]
	else:
		return [0,0,0,1]

def meanItm(input):

	if (input <= 0):
		return [1,0,0,0,0]
	elif (input < 7500):
		return [0,1,0,0,0]
	elif (input < 8000):
		return [0,0,1,0,0]
	elif (input < 8500):
		return [0,0,0,1,0]
	else:
		return [0,0,0,0,1]


def coverImg(input):

	if (input <= 0):
		return [1,0,0,0]
	elif (input < 170):
		return [0,1,0,0]
	elif (input < 185):
		return [0,0,1,0]
	else:
		return [0,0,0,1]

def coverItm(input):

	if (input <= 0):
		return [1,0,0,0]
	elif (input < 7500):
		return [0,1,0,0]
	elif (input < 8500):
		return [0,0,1,0]
	else:
		return [0,0,0,1]


def top5Img(input):

	if (input <= 0):
		return [1,0,0,0,0]
	elif (input < 165):
		return [0,1,0,0,0]
	elif (input < 180):
		return [0,0,1,0,0]
	elif (input < 200):
		return [0,0,0,1,0]
	else:
		return [0,0,0,0,1]


def top5Itm(input):

	if (input <= 0):
		return [1,0,0,0,0]
	elif (input < 7500):
		return [0,1,0,0,0]
	elif (input < 8200):
		return [0,0,1,0,0]
	elif (input < 8700):
		return [0,0,0,1,0]
	else:
		return [0,0,0,0,1]


def itmFreshness(input):

	if (input < 0.2):
		return [1,0,0,0,0]
	elif (input < 0.5):
		return [0,1,0,0,0]
	elif ( input < 0.7):
		return [0,0,1,0,0]
	elif (input < 0.9):
		return [0,0,0,1,0]
	else:
		return [0,0,0,0,1]

def authorBuyerSegment(input):
	
	options = {'A': [1,0,0,0,0,0],
	'B': [0,1,0,0,0,0],
	'C': [0,0,1,0,0,0],
	'D': [0,0,0,1,0,0],
	'E': [0,0,0,0,1,0],
	'NA':[0,0,0,0,0,1]
	}

	return options[input]

def authorSellerSegment(input):
	
	options = {'Large Merchants': [1,0,0,0,0,0],
	'Merchants': [0,1,0,0,0,0],
	'Entrepreneurs': [0,0,1,0,0,0],
	'Regulars': [0,0,0,1,0,0],
	'Occasional': [0,0,0,0,1,0],
	'NA':[0,0,0,0,0,1]
	}

	return options[input]

def authorType(input):
	
	options = {'Buyer': [1,0,0,0],
	'BuyerSeller': [0,1,0,0],
	'Seller': [0,0,1,0],
	'NA' : [0,0,0,1]
	}

	return options[input]

def collectionsAge(input):
	
	if (input <= 7):
		return [1,0,0,0,0]
	elif (input <= 45):
		return [0,1,0,0,0]
	elif (input <= 90):
		return [0,0,1,0,0]
	elif (input <= 150):
		return [0,0,0,1,0]
	else:
		return [0,0,0,0,1]

def collectionsLastUpdate(input):
	
	if (input <= 0):
		return [1,0,0,0,0]
	elif (input <= 1):
		return [0,1,0,0,0]
	elif (input <= 10):
		return [0,0,1,0,0]
	elif (input <= 60):
		return [0,0,0,1,0]
	else:
		return [0,0,0,0,1]

def os(os):
	if os in 'Windows':
		return [1,0,0,0,0,0]
	elif os in 'OS X' :
		return [0,1,0,0,0,0]
	elif os in 'iOS' :
		return [0,0,1,0,0,0]
	elif os in 'Android':
		return [0,0,0,1,0,0]
	elif os in 'Linux' :
		return [0,0,0,0,1,0]
	else:
		return [0,0,0,0,0,1]

def browser(browser):
	if browser in 'Chrome':
		return [1,0,0,0,0,0,0,0]
	elif browser in 'IE' :
		return [0,1,0,0,0,0,0,0]
	elif browser in 'Firefox' :
		return [0,0,1,0,0,0,0,0]
	elif browser in 'Safari':
		return [0,0,0,1,0,0,0,0]
	elif browser in 'Mobile Safari' :
		return [0,0,0,0,1,0,0,0]
	elif browser in 'Android Webkit' :
		return [0,0,0,0,0,1,0,0]
	elif browser in 'Opera' :
		return [0,0,0,0,0,0,1,0]
	else:
		return [0,0,0,0,0,0,0,1]

def deviceType(mobile, desktop, tab):
	l=[0,0,0]
	if (mobile in 'true'):
		l=[1,0,0]
	elif (desktop in 'true'):
		l=[0,1,0]
	elif (tab in 'true'):
		l=[0,0,1]
	else:
		l=[0,0,0]
		
	return l

#f= open("out_test")
target=[]
f= open("collections_level", "r")

count = 0
for line in f:
	l=line.split('|')
	
	#if (int(l[26])>8):
	count +=1
	#if (int(l[3]) < 250):

	#if count > 100:
	#	break
	#temp = [int(i) for i in l[33:40]]
	target.append(float(l[1]))
	
	if (float(l[1])):
		Y.append(1)
	else:
		Y.append(0)
	
	temp =[]
	
	for i in (6,7,8,16,19,20,21,22):
		temp.append(float(l[i]))
	
	

	a = authorBuyerSegment(l[23])
	for x in a:
		temp.append(x)

	a = authorSellerSegment(l[24])
	for x in a:
		temp.append(x)

	a = authorType(l[25])
	for x in a:
		temp.append(x)


	# Collections age and last update
	
	temp.append(float(l[26]))
	
	temp.append(float(l[27]))


	#Item counts
	temp.append(float(l[28]))
	#page count 
	temp.append(float(l[34]))
	# Is owner
	temp.append(float(l[35]))

	#temp.append(int(l[36]))

	

	if (float(l[38]) or float(l[39]) or float(l[40]) or float(l[41]) or float(l[42]) or float(l[43])):
		temp.append(1)
	else:
		temp.append(0)
		
	
	for i in range(44,46):
		
		#temp.append(int(l[i]))
		if (float(l[i])):
			temp.append(1)
		else:
			temp.append(0)
	
	#temp.append(int(l[46]))
	#temp.append(int(l[47]))

	"""
	a= int(l[49])
	
	if (a):
		if (a <15):
			temp.append(1)
			
		else:
			temp.append(0)
			
	else:
		temp.append(0)
	
	"""


	X.append(temp)
		

print("phase 1 loaded")

f.close()




print "Y counter:", Counter(Y)
print "Target counter", Counter(target)
print np.mean(target)
print "transforming to imp. features"

#clf = ExtraTreesClassifier(n_estimators=25)
#X = clf.fit(X,Y).transform(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split( 
	X, Y, test_size=0.2, random_state=0)

del X,Y
#X_mob_train, X_mob_test, y_mob_train, y_mob_test = cross_validation.train_test_split( 
#	X_mob, Y_mob, test_size=0.3, random_state=0)

#print "train size, test size", len(X_train)/1000., len(X_test)/1000.


clfDT = tree.DecisionTreeClassifier()

clfRF = RandomForestClassifier()

#clfPerc = linear_model.Perceptron()


clfGNB = GaussianNB()
#clfGradBoo = GradientBoostingClassifier()


clfLR = linear_model.LogisticRegression(penalty='l1', dual=False)
clfSGD = SGDClassifier(loss="hinge", penalty="l2")
clfAdaB = AdaBoostClassifier()

#clf = linear_model.LassoLars(alpha=.1)
#clf = linear_model.Ridge (alpha = .5)
#clfLinear = linear_model.LinearRegression()
#clf = svm.SVC(C=1.0, kernel='rbf')

#clfSGD = SGDClassifier(loss="hinge", penalty="l2")
#clf = SGDClassifier(loss="log").fit(X, y)
	#clf.coef_
#clfGP=gaussian_process.GaussianProcess()


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

del scaler


print "--------------printing SGDClassifier for BBOWA for combo without Follow and purchase----------------"
start = time.time()

clf=clfSGD.fit(X_train,y_train)

print "coefficients:", clf.coef_

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"


print "--------------printing GaussianNB for BBOWA for combo without Follow and purchase----------------"
start = time.time()

clf=clfGNB.fit(X_train,y_train)

#print "coefficients:", clf.coef_

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"


print "--------------printing DecisionTreeClassifier for BBOWA for combo using BBOWA, follow, registeration ----------------"
start = time.time()

clf=clfDT.fit(X_train,y_train)

#print "coefficients:", clf.coef_

print ("feature_importances_", clf.feature_importances_)

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"



print "--------------printing LogisticRegression for for BBOWA for combo without Follow and purchase----------------"
start = time.time()

clf=clfLR.fit(X_train,y_train)

print "coefficients:", clf.coef_

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"



print "--------------printing RandomForestClassifier  for BBOWA for combo without Follow and purchase----------------"
start = time.time()

clf=clfRF.fit(X_train,y_train)

#print "coefficients:", clf.coef_

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

print ("feature_importances_", clf.feature_importances_)
#print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"


print "--------------printing AdaBoostClassifier for BBOWA for combo without Follow and purchase ----------------"
start = time.time()

clf=clfAdaB.fit(X_train,y_train)

#print "coefficients:", clf.coef_

print ("feature_importances_", clf.feature_importances_)

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

#print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"

"""
print "--------------printing DecisionTreeClassifier for BBOWA for combo using BBOWA, follow, registeration ----------------"
start = time.time()

clf=clfDT.fit(X_train,y_train)

#print "coefficients:", clf.coef_

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

#print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"

print "--------------printing RandomForestClassifier  for BBOWA for combo using BBOWA, follow, registeration----------------"
start = time.time()

clf=clfRF.fit(X_train,y_train)

#print "coefficients:", clf.coef_

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

#print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"





"""

"""
print "--------------printing GradientBoostingClassifier for VI for combo using BBOWA, follow, registeration----------------"
start = time.time()

clf=clfGradBoo.fit(X_train,y_train)

#print "coefficients:", clf.coef_

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

#print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

#print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred.to, target_names=target_names))

del clf,y_pred
end = time.time()

print "execution time:", (end - start), "s"

"""


"""
scaler = StandardScaler()
scaler.fit(X_mob_train)
X_mob_train = scaler.transform(X_mob_train)

X_mob_test = scaler.transform(X_mob_test)

target_names = ['class 0', 'class 1']
"""
#X=preprocessing.normalize(X, norm='l2')
#X_test=preprocessing.normalize(X_test, norm='l2')
#X_train=preprocessing.normalize(X_train, norm='l2')
#min_max_scaler = preprocessing.MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)
#X_test = min_max_scaler.transform(X_test)
#print y_out

#precision, recall, thresholds = precision_recall_curve(y_test,y_pred)

#print precision

#print recall

#print thresholds

"""
print "-----------------Tree based feature selection-------------"

clf = ExtraTreesClassifier()
start = time.time()
X_new = clf.fit(X,Y).transform(X)
print clf.feature_importances_
print X_new.shape

end = time.time()

print "execution time:", (end - start), "s"
del X_new,clf


print "----------------------Feature selection using RandomizedPCA--------------"

pca = RandomizedPCA()
start = time.time()
pca.fit(X)
print(pca.explained_variance_ratio_)
end = time.time()

print "execution time:", (end - start), "s"

"""
"""
print "printing decision trees"
start = time.time()

clf = clf.fit(X_train, y_train)

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

print clf.score(x1_new, y1_new) #check it corerct it, its wrong

print clf.score(X_test,y_test)

print "r square ", r2_score(y_test, y_pred)

end = time.time()



print "printing GaussianNB"
start = time.time()

clf=clf2.fit(X_train,y_train)

y_pred = [clf.predict(x) for x in X_test]

print (precision_recall_fscore_support(y_test, y_pred, average=None))

print "r square ", r2_score(y_test, y_pred)

print clf.score(x1_new, y1_new) #check it corerct it, its wrong

print clf.score(X_test,y_test)


end = time.time()

print "execution time:", (end - start), "s"
#print clf.predict_log_proba(X_test)

#print clf.predict_proba(X_test)



"""
"""
print "---------------printing LR with mobile,tab---------------"
start = time.time()

clf=clfLR.fit(X_mob_train,y_mob_train)

#print "coefficients:", clf.coef_

y_mob_pred = [clf.predict(x) for x in X_mob_test]

print (precision_recall_fscore_support(y_mob_test, y_mob_pred, average=None))

print "r square ", r2_score(y_mob_test, y_mob_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

print clf.score(X_mob_test,y_mob_test)

print "roc curve", roc_auc_score(y_mob_test,y_mob_pred)

print "accuracy_score", accuracy_score(y_mob_test, y_mob_pred)


#print ("performance of classes: ", "\n", classification_report(y_test, y_pred, target_names=target_names))

end = time.time()

print "execution time:", (end - start), "s"

"""
"""

print "--------------printing Linear regression----------------"
start = time.time()

clf=clfLinear.fit(X_train,y_train)

#print "coefficients:", clf.coef_

y_pred = [clf.predict(x) for x in X_test]

#print (precision_recall_fscore_support(y_test, y_pred, average=None))

print "r square ", r2_score(y_test, y_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

print clf.score(X_test,y_test)

print "roc curve", roc_auc_score(y_test,y_pred)

#print "accuracy_score", accuracy_score(y_test, y_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred, target_names=target_names))

end = time.time()

print "execution time:", (end - start), "s"


print "---------------printing Linear regression with mobile,tab---------------"
start = time.time()

clf=clfLinear.fit(X_mob_train,y_mob_train)

#print "coefficients:", clf.coef_

y_mob_pred = [clf.predict(x) for x in X_mob_test]

#print (precision_recall_fscore_support(y_mob_test, y_mob_pred, average=None))

print "r square ", r2_score(y_mob_test, y_mob_pred)

#print clf.score(x1_new, y1_new) #check it corerct it, its wrong

print clf.score(X_mob_test,y_mob_test)

print "roc curve", roc_auc_score(y_mob_test,y_mob_pred)

#print "accuracy_score", accuracy_score(y_mob_test, y_mob_pred)

#print ("performance of classes: ", "\n", classification_report(y_test, y_pred, target_names=target_names))

end = time.time()

print "execution time:", (end - start), "s"
"""	