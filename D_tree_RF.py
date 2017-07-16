    
import numpy as np
from collections import defaultdict,Counter
import random
from scipy.stats.mstats import mode
from numpy import nan
from sklearn.metrics import precision_recall_curve
from random import randint
from sklearn.metrics import confusion_matrix



class Node:
    def __init__(s,f,pos,l):
        s.left=None
        s.right=None
        s.f=f
        s.pos=pos
        s.l=l

def parse_data():

    data=np.genfromtxt('./data/binary_data.csv',delimiter=',',dtype=float)[:,1:]
    y=np.genfromtxt('./data/binary_data.csv',delimiter=',',dtype=str)[:,0]
    labels,y=np.unique(y,return_inverse=True)
    dataset=np.hstack((data,y[:,np.newaxis]))
    random.shuffle(dataset)
    return dataset,list(labels)

def get_dataset_splits(data,labels):
    datasets=[0]*10
    for label in xrange(len(labels)):
        if label==0:
            x=data[data[:,-1]==label]
            random.shuffle(x)
            for i,split in enumerate(np.array_split(x,10)):
                datasets[i]=split
        else:
            x=data[data[:,-1]==label]
            random.shuffle(x)
            for i,split in enumerate(np.array_split(x,10)):
                datasets[i]=np.vstack((datasets[i],split))

    return datasets

def get_test_train(datasets):
    
    for i in xrange(len(datasets)):
        testset=datasets[i]
        trainset=None
        for j in xrange(len(datasets)):
            if j!=i:
                if trainset==None:
                    trainset=datasets[j]
                else:
                    
                    try:
                        trainset=np.vstack((trainset,datasets[j]))
                    except ValueError, e:
                        print "error in vstack"
                        print trainset
                        

        yield testset,trainset
    

def find_majority_class(labels):
    return Counter(labels).most_common()[0][0]
    

def is_missing(val):
    
    if val == 11111111111:
        print "yo"
        return True
    else:
        return False

def get_info_gain(data):
    info_gains=[]
    
    datasets_temp = list(data)
    for val in np.unique(data[:,0]):
        
        mean = np.mean(data[:,0])

        
        a=np.argmax(data[:,0], axis=0)
        data[a,0] = mean


        g1=data[data[:,0]<=val][:,-1]
        g2=data[data[:,0]>val][:,-1]
        if g1.shape[0]!=0:
            ent1=(float)(g1.shape[0])*calc_info_gain(g1)/(data.shape[0])
        else:
            ent1=0
        if g2.shape[0]!=0:
            ent2=(float)(g2.shape[0])*calc_info_gain(g2)/(data.shape[0])
        else:
            ent2=0
        avg_ent=ent1+ent2
        info_gains.append((avg_ent,val))
    return min(info_gains)

def calc_info_gain(g):
    term=(float)(1)/(g.shape[0])
    gain=0
    for label,count in Counter(g).most_common():
        if count==0:
            continue
        gain-=count*term*np.log2(count*term)
    return gain



def tree_predict(tree,x,labels):
    #print labels
    l = [i for i in x if i != 11111111111]
    mean = np.mean(l)
    for i in range(len(x)):
        if x[i] == 11111111111:
            x[i] = mean

    while(tree not in list(labels)):
        
        f=tree.f
        split=tree.pos
        if x[f]<=split:
            tree=tree.left
        else:
            tree=tree.right
    return tree

def get_best_split(data):
    info_gains=[]
    for f in xrange(data.shape[1]-1):
        f_ent,val=get_info_gain(data[:,[f,-1]])
        info_gains.append((f_ent,f,val))

    return min(info_gains) 

def build_tree(root,data,level,datasize):

    if data.shape[0]==1:
        return find_majority_class(data[:,-1])

    if np.unique(data[:,-1]).shape[0]==1:
        return int(data[0,-1])

    f_ent,imp_f,val=get_best_split(data)
    root=Node(imp_f,val,level)
    
    d1=data[data[:,imp_f]<=val]
    d2=data[data[:,imp_f]>val]

    root.left=build_tree(root.left,d1,level+1,datasize)
    root.right=build_tree(root.right,d2,level+1,datasize)
    return root
    
def calculate_accs(true,predicted,labels,confusion):
    count=0
    for i,j in zip(true,predicted):
        confusion[labels[int(i)]][labels[int(j)]]+=1
        if i==j: count+=1
    return (float)(count)/len(true)

def get_acc_RF(data,tree,labels,confusion):
    predictions=[]
    a = tree_predict(tree,data,range(len(labels)))
    predictions.append(a)

    return predictions


def get_acc(data,tree,labels,confusion):
    predictions=[]
    for i in xrange(data.shape[0]):
        predictions.append(tree_predict(tree,data[i,:],range(len(labels))))
    acc= calculate_accs(data[:,-1],predictions,labels,confusion)
    precision, recall, thresholds = precision_recall_curve(data[:,-1],predictions)
    
    return (acc, precision, recall)


def print_confusion(x,labels):
    with open('dt_confusion.txt','w') as fil:
        print 'CLASS '+' '.join(labels)
        for l1 in labels:
            print l1+'  ',
            for l2 in labels:
                print "%2d " % x[l1][l2],
                fil.write("%2d," %x[l1][l2])
            fil.write('\n')
            print '\n'

def print_precision_recall():
    fl = open('dt_confusion.txt')
    fl = fl.readlines()
    res = []
    for line in fl:
        l = line.split(',')
        res.append(float(l[0]))
        res.append(float(l[1]))
    a = res[0]
    b = res[1]
    c = res[2]
    d = res[3]
    print a, b, c, d

    print "precision: ", d/float(c+d)
    print "recall: ", d/float(b+d)

def decisionTree(generator, labels):
    fold_accs=[]
    fold_prec = []
    fold_recall = []
    conf_matr={}
    for l1 in labels:
        conf_matr[l1]=defaultdict(int)

    for i,(test,train) in enumerate(generator):
        root=None
        datasize=train.shape[0]
        tree=build_tree(root,train,1,datasize)
        test_acc, test_precision, test_recall =get_acc(test,tree,labels,conf_matr)
        print "Processing Fold %d..." %(i+1)
        fold_accs.append(test_acc)
        fold_prec.append(test_precision)
        fold_recall.append(test_recall)

    return fold_accs, fold_prec, fold_recall, conf_matr

def decisionTreeForRF(train, test, labels):
    predicted = []    
    conf_matr={}
    for l1 in labels:
        conf_matr[l1]=defaultdict(int)

    root=None
    datasize=train.shape[0]
    tree=build_tree(root,train,1,datasize)
    for x in test:
        predicted.append(get_acc_RF(x, tree,labels,{}))

    return predicted
    

def randomForest(datasets, labels):
    
    test_predictions = []
    total_test_data = []


    test_data = datasets[-1]
    train_data = datasets[:-1]

    for i in train_data:
        test_predictions.append(decisionTreeForRF(i,test_data, labels))

    final_predictions = []

    l= len(test_predictions[0])
    for i in range(l):
        temp = []
        for k in test_predictions:
            temp.append(k[i][0])
            #print k[i]
        final_predictions.append(temp)

    test_predictions = []
    for i in final_predictions:
        test_predictions.append(Counter(i).most_common()[0][0])


    
    conf_matr={}
    for l1 in labels:
        conf_matr[l1]=defaultdict(int)
    acc = calculate_accs(test_data[:,-1] ,test_predictions,labels,conf_matr)

    precision, recall, thresholds = precision_recall_curve(test_data[:,-1],test_predictions)
    cm = confusion_matrix(test_data[:,-1],test_predictions)
    print "confusion matrix", cm
    
    return (acc, precision, recall)


def randomForest_10fold(datasets, labels):
    fold_accs = []
    fold_prec = []
    fold_recall = []
    for k in range(len(datasets)):
        print "Processing Fold %d..." %(k+1)
        test_predictions = []
        total_test_data = []
        if k == 0:
            test_data = datasets[k]
            train_data = datasets[k+1:]

        else:
            test_data = datasets[k]
            train_data = datasets[:k] + datasets[k+1:]

        datasets_temp = train_data + [test_data]
        acc, precision, recall = randomForest(datasets_temp, labels)
        fold_accs.append(acc)
        fold_prec.append(precision)
        fold_recall.append(recall)


    return (fold_accs, fold_prec,fold_recall)

def removeRandomData():
    fl = open('data_banknote_authentication.txt')
    data = []
    for line in fl:
        l = line.split(',')
        temp = []

        data.append([float(i) for i in l[:-1]])
        

    y=np.genfromtxt('data_banknote_authentication.txt',delimiter=',',dtype=str)[:,-1]

    k = random.sample(range(1, 100), 50)

    for i in k:
        p = randint(0,3)
        data[i][p] = 11111111111

    labels,y=np.unique(y,return_inverse=True)
    dataset=np.hstack((data,y[:,np.newaxis]))
    
    return dataset,list(labels)

def main(model_type):
    

    if model_type == 1:

        data,labels=parse_data()
        datasets=get_dataset_splits(data,labels)
        generator=get_test_train(datasets)
        fold_accs=[]
        fold_prec = []
        fold_recall = []

        conf_matr={}
        for l1 in labels:
            conf_matr[l1]=defaultdict(int)


        for i,(test,train) in enumerate(generator):
            root=None
            datasize=train.shape[0]
            tree=build_tree(root,train,1,datasize)
            test_acc, test_precision, test_recall =get_acc(test,tree,labels,conf_matr)
            print "Processing Fold %d..." %(i+1)
            #print test_precision.shape, test_recall.shape
            #fold_accs.append(test_acc)
            #fold_prec.append(test_precision)
            #fold_recall.append(test_recall)

        print '\nAccuracy: ',np.average(fold_accs)
        print '\nConfusion Matrix: '
        label_order=['1','2','3']
        print_confusion(conf_matr,label_order)
        print_precision_recall()
            #return conf_matr,label_orders
            
    elif model_type == 2:

        data,labels=parse_data()
        datasets=get_dataset_splits(data,labels)

        fold_accs, fold_prec, fold_recall = randomForest_10fold(datasets, labels)
        print np.mean(fold_accs)
        precision = []
        recall = []
        for i in fold_prec:
            a = np.mean(i[0:])
            precision.append(a)
        for i in fold_recall:
            a = np.mean(i[:-1])
            recall.append(a)
        print np.mean(precision)
        print np.mean(recall)

    elif model_type == 3:
        data,labels=removeRandomData()
        datasets=get_dataset_splits(data,labels)
        generator=get_test_train(datasets)
        fold_accs=[]
        fold_prec = []
        fold_recall = []

        conf_matr={}
        for l1 in labels:
            conf_matr[l1]=defaultdict(int)


        for i,(test,train) in enumerate(generator):
            root=None
            datasize=train.shape[0]
            tree=build_tree(root,train,1,datasize)
            test_acc, test_precision, test_recall =get_acc(test,tree,labels,conf_matr)
            print "Processing Fold %d..." %(i+1)
            #print test_precision.shape, test_recall.shape
            #fold_accs.append(test_acc)
            #fold_prec.append(test_precision)
            #fold_recall.append(test_recall)
        print '\nAccuracy: ',np.average(fold_accs)
        print '\nConfusion Matrix: '
        label_order=['1','2','3']
        print_confusion(conf_matr,label_order)
        print_precision_recall()
            #return conf_matr,label_orders

    elif model_type == 4:

        data,labels=removeRandomData()
        datasets=get_dataset_splits(data,labels)
        fold_accs, fold_prec, fold_recall = randomForest_10fold(datasets, labels)
        print np.mean(fold_accs)
        precision = []
        recall = []
        for i in fold_prec:
            a = np.mean(i[0:])
            precision.append(a)
        for i in fold_recall:
            a = np.mean(i[:-1])
            recall.append(a)
        print np.mean(precision)
        print np.mean(recall)
    
    
    


main(1)
