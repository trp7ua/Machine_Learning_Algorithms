import numpy as np
from numpy import *
import random
import os
from PIL import Image

class cluster_node:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None,count=1):
        self.left=left
        self.right=right
        self.vec=vec
        self.id=id
        self.distance=distance
        self.count=count #only used for weighted average 

def L2dist(v1,v2):
    return sqrt(sum((v1-v2)**2))

def hcluster(features,distance=L2dist):
    #cluster the rows of the "features" matrix
    distances={}
    currentclustid=-1

    # clusters are initially just the individual rows
    clust=[cluster_node(array(features[i]),id=i) for i in range(len(features))]

    while len(clust)>1:
        lowestpair=(0,1)
        closest=distance(clust[0].vec,clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id,clust[j].id) not in distances: 
                    distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)

                d=distances[(clust[i].id,clust[j].id)]

                if d < closest:
                    closest=d
                    lowestpair=(i,j)

        # calculate the average of the two clusters
        mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 \
            for i in range(len(clust[0].vec))]

        # create the new cluster
        newcluster=cluster_node(array(mergevec),left=clust[lowestpair[0]],
                             right=clust[lowestpair[1]],
                             distance=closest,id=currentclustid)

        # cluster ids that weren't in the original set are negative
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]

def drawnode(draw,clust,x,y,scaling,imlist,img):
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        # Line length
        ll=clust.distance*scaling
        # Vertical line from this cluster to children    
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))    

        # Horizontal line to left item
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))    

        # Horizontal line to right item
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))        

        # Call the function to draw the left and right nodes    
        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,imlist,img)
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,imlist,img)
    else:   
        # If this is an endpoint, draw a thumbnail image
        nodeim = Image.open(imlist[clust.id])
        nodeim.thumbnail((20,20))
        ns = nodeim.size
        img.paste(nodeim,(x,y-ns[1]//2,x+ns[0],y+ns[1]-ns[1]//2))

def drawdendrogram(clust,imlist,jpeg='clusters.jpg'):
    # height and width
    h=getheight(clust)*20
    w=1200
    depth=getdepth(clust)

    # width is fixed, so scale distances accordingly
    scaling=float(w-150)/depth

    # Create a new image with a white background
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    draw.line((0,h/2,10,h/2),fill=(255,0,0))    

    # Draw the first node
    drawnode(draw,clust,10,(h/2),scaling,imlist,img)
    img.save(jpeg)   

def extract_clusters(clust,dist):
    # extract list of sub-tree clusters from hcluster tree with distance < dist
    clusters = {}
    if clust.distance < dist:
        # we have found a cluster subtree
        return [clust]
    else:
        # check the right and left branches
        cl = []
        cr = []
        if clust.left!=None: 
            cl = extract_clusters(clust.left,dist=dist)
        if clust.right!=None: 
            cr = extract_clusters(clust.right,dist=dist)
        return cl+cr


def get_cluster_elements(clust):
    # return ids for elements in a cluster sub-tree
    if clust.id>0:
        # positive id means that this is a leaf
        return [clust.id]
    else:
        # check the right and left branches
        cl = []
        cr = []
        if clust.left!=None: 
            cl = get_cluster_elements(clust.left)
        if clust.right!=None: 
            cr = get_cluster_elements(clust.right)
        return cl+cr 

def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    
    return X

X = init_board(10)

tree = hcluster(X)

drawdendrogram(tree,imlist,jpeg='sunset.jpg')