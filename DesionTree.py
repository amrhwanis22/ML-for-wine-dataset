import pandas as ps
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def readFile():
    file=ps.read_csv('Wine_data.csv')
    X=file.iloc[:,0:11]
    Y=file.iloc[:,11]
    print(file)
    return X,Y

X,Y=readFile()

print('XXXXXXXXX',X)
print('YYYYYYYYYYYY',Y)
KNN=KNeighborsClassifier()
Naive=GaussianNB()

Tree=tree.DecisionTreeClassifier()

xtrin,xtest,ytrain,ytest=train_test_split(X,Y,random_state=4,test_size=0.2)
Tree.fit(xtrin,ytrain)
KNN.fit(xtrin,ytrain)
predKNN=KNN.predict(xtest)
print(Tree.predict((xtest)))
Ored=Tree.predict(xtest)
print(predKNN)
Naive.fit(xtrin,ytrain)
print(Naive.predict(xtest))
nn=Naive.predict(xtest)
print ('accScore',accuracy_score(ytest,Ored))
print ('accScore',accuracy_score(ytest,predKNN))
print ('accScore',accuracy_score(ytest,nn))
