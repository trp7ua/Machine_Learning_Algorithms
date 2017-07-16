import numpy as np
import random
from collections import defaultdict,Counter
from scipy.stats.mstats import mode
from numpy import nan
from sklearn.metrics import precision_recall_fscore_support

class Node:
    def __init__(s,f,pos,l):
        s.left=None
        s.right=None
        s.f=f
        s.pos=pos
        s.l=l

def parse_data():
    #data=np.genfromtxt('data_banknote_authentication.txt',delimiter=',',dtype=float,invalid_raise=False,
    #            missing_values='',
    #            usemask=False,
    #            filling_values=1111111111)[:,:-1]
    data=np.genfromtxt('./data/binary_data.csv',delimiter=',',dtype=float)[:,1:]
    y=np.genfromtxt('./data/binary_data.csv',delimiter=',',dtype=str)[:,0]
    labels,y=np.unique(y,return_inverse=True)
    dataset=np.hstack((data,y[:,np.newaxis]))

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
                    trainset=np.vstack((trainset,datasets[j]))
        yield testset,trainset


def find_majority_class(labels):
    return Counter(labels).most_common()[0][0]

def get_info_gain(data):
    info_gains=[]
    g1 = []
    g2 = []
    for val in np.unique(data[:,0]):

        m = mode(data[:,0])
        m = float(m[1])
        n = len(data[:,0])
        p = dict(Counter(data[:,0]))
        m = 0
        for i in p:
            m += p[i]*i
        m = m/float(n)
        
        for i in range(len(data[:,0])):
            if is_missing(data[i,0]):
                print "yo"
                data[i,0] = m


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

def is_missing(val):
    #print val
    if str(val) in '1111111111':
        return True
    else:
        return False


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


def get_acc(data,tree,labels,confusion):
    predictions=[]
    for i in xrange(data.shape[0]):
        predictions.append(tree_predict(tree,data[i,:],range(len(labels))))
    acc= calculate_accs(data[:,-1],predictions,labels,confusion)
    return acc


def get_acc_RF(data,tree,labels,confusion):
    predictions=[]
    a = tree_predict(tree,data,range(len(labels)))
    predictions.append(a)

    return predictions


def print_confusion(x,labels):
    print 'CLASS '+' '.join(labels)
    for l1 in labels:
        print l1+'  ',
        for l2 in labels:
            #print l1, l2
            #print x
            print "%2d " % x[l1][l2],
        print '\n'

def decisionTree(generator, labels):
    fold_accs=[]
    conf_matr={}
    for l1 in labels:
        conf_matr[l1]=defaultdict(int)

    for i,(test,train) in enumerate(generator):
        root=None
        datasize=train.shape[0]
        tree=build_tree(root,train,1,datasize)
        test_acc=get_acc(test,tree,labels,conf_matr)
        print "Processing Fold %d..." %(i+1)
        fold_accs.append(test_acc)

    return fold_accs, conf_matr

def decisionTreeForRF(train, test, labels):
    predicted = []    
    root=None
    datasize=train.shape[0]
    tree=build_tree(root,train,1,datasize)
    for x in test:
        predicted.append(get_acc_RF(x, tree,labels,{}))
    print len(predicted), len(test)

    return predicted
    

def randomForest(datasets, labels):
    
    test_predictions = []
    total_test_data = []


    test_data = datasets[-1]
    train_data = datasets[:-1]

    for i in train_data:
        #print "---"
        test_predictions.append(decisionTreeForRF(i,test_data, labels))

    final_predictions = []

    l= len(test_predictions[0])
    #print l
    for i in range(l):
        temp = []
        for k in test_predictions:
            temp.append(k[i][0])
            #print k[i]
        final_predictions.append(temp)

    test_predictions = []
    for i in final_predictions:
        test_predictions.append(Counter(i).most_common()[0][0])

    #print "len of test_predictions: ", len(test_predictions)

    
    conf_matr={}
    for l1 in labels:
        conf_matr[l1]=defaultdict(int)
    acc= calculate_accs(test_data[:,-1] ,test_predictions,labels,conf_matr)

    return acc


def randomForest_10fold(datasets, labels):
    acc = []
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
        acc.append(randomForest(datasets_temp, labels))

    print np.mean(acc)


def main(model_type):
    data,labels=parse_data()
    datasets=get_dataset_splits(data,labels)
    #print len(datasets)

    generator=get_test_train(datasets)
    fold_accs=[]

    conf_matr={}
    for l1 in labels:
        conf_matr[l1]=defaultdict(int)
    #randomForest(datasets, labels)
    if model_type == 1:
        for i,(test,train) in enumerate(generator):
            root=None
            datasize=train.shape[0]
            tree=build_tree(root,train,1,datasize)
            test_acc=get_acc(test,tree,labels,conf_matr)
            print "Processing Fold %d..." %(i+1)
            fold_accs.append(test_acc)

    elif model_type == 2:
        randomForest_10fold(datasets, labels)
    #fold_accs, conf_matr =  decisionTree(generator, labels)
    
    

    print '\nAccuracy: ',np.average(fold_accs)
    print '\nConfusion Matrix: '
    label_order=['1','2']
    print_confusion(conf_matr,label_order)
    return conf_matr,label_order


conf_matr,labels=main(1)
