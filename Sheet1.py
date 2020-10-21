"""
Solution to Sheet 1 of the Machine Learning I: Foundations Course
    
Author: Billy Joe Franks
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from functools import partial
import collections
from random import shuffle

xmin=160
xmax=185
ymin=40
ymax=85

"this function extracts the data as a list of lists of strings."
"Conversion to the appropriate datatype occurs whenever it is needed."
def extractData(file):
    data = []
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',',quotechar='|')
        for row in spamreader:
            data.append(row)
    return data

"function to plot multiple lines. Used for exercise 3 and 4"
def plotline(data,lines,file):
    plt.scatter([float(row[1]) for row in data],[float(row[2]) for row in data],c=[int(row[3]) for row in data])
    "this fixes the plotted region"
    plt.axis((xmin,xmax,ymin,ymax))
    "this draws all input lines"
    for line in lines:
        plt.plot(line[0],line[1])
    "this save the plot to the harddrive for inclusion in a pdf"
    plt.savefig(file)
    plt.close()

"distance between two points according to the chosen data structure. Used to sort the points in KNN for prediction"
def distance_to_point(item, point):
    return np.linalg.norm(np.array(item['po'])-np.array(point))

"this is the k nearest neighbour classifier"
class KNN(object):

    "this allows us to check wether our classifier was trained"
    def __init__(self):
        self.data=None
        self.k=None

    "training a k nearest neighbour classifier means remembering the data. We also include the choice of k as training"
    def train(self,data,k):
        self.data=data
        self.k=k

    "predict a list of inputs"
    def predict(self,dataX):
        if self.k==None or self.data==None:
            raise("Model not trained!")
        dataY = []
        for x in dataX:
            dataY.append(self.predictPoint(x))
        return dataY

    "predict a single input"
    def predictPoint(self,x):
        "We define a function used for sorting all datapoints according to the distance"
        comp = partial(distance_to_point, point=x)
        "We sort all datapoints according to the distance"
        dataSorted=sorted(self.data, key=comp)
        "We consider the closest k points and extract their classes"
        classes=[datum['cl'] for datum in dataSorted[:self.k]]
        "We count the occurences of each class and used the most frequent class for prediction"
        counts=collections.Counter(classes)
        return counts.most_common(1)[0][0]

"This function does t-times k-fold cross validation with knn parameter knnk."
def CV(t,k,training_data,knnk):
    accs = []
    for _ in range(0,t):
        "Shuffle the data, this results in a random choice for each fold."
        shuffle(training_data)
        "Split the folds into k foldsize sized chunks"
        foldsize=int(len(training_data)/k)
        folds=[training_data[(i*foldsize):(i*foldsize)+foldsize] for i in range(0,k)]
        for j in range(0,k):
            "choose the j-th fold as testset and the rest of the folds as training set."
            training=sum(folds[:j],[])+sum(folds[j+1:],[])
            test=folds[j]
            "use knn to predict all the points in the testset"
            knn=KNN()
            knn.train(training,knnk)
            predictions=knn.predict([point['po'] for point in test])
            correct_predictions=[point['cl'] for point in test]
            "calculate the accuracy"
            accuracy=[p==c for p,c in zip(predictions,correct_predictions)]
            accs.append(accuracy.count(True)/len(test))
    "return the average accuracy"
    return sum(accs)/len(accs)

if __name__ == "__main__":
    data=extractData('DWH_Training.csv')

    "Exercise 3a)"
    plotline(data,[],'plot0.pdf')

    "Exercise 3b)"
    x= np.linspace(xmin,xmax)
    y= [68]*len(x)
    plotline(data,[[x,y]],'plot1.pdf')

    "Exercise 3c)"
    x= np.linspace(xmin,xmax)
    y= [62]*len(x)
    plotline(data,[[x,y]],'plot2.pdf')

    "Exercise 3d)"
    y= np.linspace(ymin,ymax)
    x= [171]*len(y)
    plotline(data,[[x,y]],'plot3.pdf')

    "Exercise 3e)"
    y= np.linspace(ymin,ymax)
    x= [181]*len(y)
    plotline(data,[[x,y]],'plot4.pdf')

    "Exercise 3f)"
    x=[169,175]
    y=[85,40]
    plotline(data,[[x,y]],'plot5.pdf')

    "Exercise 3g)"
    lines=[]
    x=[169,175]
    y=[85,40]
    lines.append([x,y])
    
    y= np.linspace(40,85)
    x= [171]*len(y)
    lines.append([x,y])
    
    x= np.linspace(162,178)
    y= [68]*len(x)
    lines.append([x,y])
    
    plotline(data,lines,'plot6.pdf')

    "Exercise 4a)"
    w=np.array([0.576,0.047])
    bs=[-103,-102,-101,-100,-99]
    x=np.linspace(160,180)
    lines=[]
    for b in bs:
        y=-(w[0]*x+b)/w[1]
        lines.append([x,y])
    plotline(data,lines,'plot7.pdf')

    "Exercise 4b)4c)4d)"
    bacc = []
    for b in bs:
        errors = 0
        for row in data:
            p=np.array([float(row[1]),float(row[2])])
            "Exercise 4b)"
            distance = (w.T@p+b)/np.linalg.norm(w)
            "Exercise 4c)"
            y=(-1 if (distance > 0) else 1)
            if y!=int(row[3]):
                errors+=1
        bacc.append(1-(errors/len(data)))
    "Exercise 4c)"
    print("The accuracies of the different hyperplanes are: "+str(bacc))

    "Exercise 4e)"
    bestacc=bacc[1]
    dataTest=extractData('DWH_test.csv')
    errors = 0
    for row in dataTest:
        p=np.array([float(row[1]),float(row[2])])
        y=(-1 if ((w.T@p+bs[1])/np.linalg.norm(w) > 0) else 1)
        if y!=int(row[3]):
            errors+=1
    testacc=1-(errors/len(dataTest))
    print("The best accuracy was: "+str(bestacc))
    print("The test accuracy of this hyperplane is: "+str(testacc))

    "Exercise 5"
    knn = KNN()
    training = []
    for row in data:
        point=dict()
        point['po']=[float(row[1]),float(row[2])]
        point['cl']=int(row[3])
        training.append(point)

    accs = dict()
    for k in [3,5,20]:
        accs[k]=CV(1,10,training,k)
    print(accs)
