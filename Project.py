# -*- coding: utf-8 -*-


import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #x train x test , y train y test we will have.
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,classification_report
import seaborn as sns



train_data=pd.read_csv("train.csv")



test_data=pd.read_csv("test.csv")

test_data

train_data

test_data

train_data.info()

test_data.shape #info for our data (row and column)

test_data.info()

train_data.describe()

train_data.plot(x='price_range', y='ram' , kind='scatter') #range with respect to ram(scatter is shape of lines)
plt.show()
#result: The higher the ram, the higher the price.

train_data.plot(x='price_range',y='fc',kind='scatter') #price range wrt front camera
plt.show()

train_data.plot(x='price_range',y="battery_power",kind='scatter') #price range wrt battery power
plt.show()

train_data.plot(x='price_range',y='fc',kind='scatter') 
plt.show()  #price range wrt front camera

train_data.plot(x='price_range',y='n_cores',kind='scatter')
plt.show()

train_data.plot(x='price_range',y='n_cores',kind='scatter')
plt.show()

train_data.isnull().sum()

train_data.plot(kind='box',figsize=(20,10))
plt.show()
 
#histogram
plt.hist(train_data["ram"], bins =10, color="orange")
plt.title("Ram Distribution")
plt.xlabel("Ram")
plt.ylabel("Value")
plt.show()

plt.hist(train_data["price_range"], bins =10, color="blue")
plt.title("price_range Distribution")
plt.xlabel("price_range")
plt.ylabel("Value")
plt.show()


X=train_data.drop('price_range',axis=1) #deleting price_range column

test_data=test_data.drop('id',axis=1) #delete id column

test_data.head()

test_data.shape

Y=train_data['price_range']


#correlations between attrıbutes 
fig = plt.subplots (figsize = (14, 14))
sns.heatmap(train_data.corr (), square = True, cbar = True, annot = True, cmap="GnBu", annot_kws = {'size': 8})
plt.title('Correlations between Attributes')
plt.show ()

#preprocessing
std=StandardScaler()

X_std=std.fit_transform(X) #analyze all patterns in data and transform
test_data_std=std.transform(test_data)

X_std #new array

test_data_std


# #Training The Model
# First Algorithm → Decision Tree

dt=DecisionTreeClassifier()
dt.fit(X_std,Y) #Training
dt.predict(test_data_std)
test_data

# Second Algorithm → KNN (K-Nearest Neighbor)

knn=KNeighborsClassifier()
knn.fit(X_std,Y)
knn.predict(test_data_std)
test_data

# Third Algorithm → Logistic Regression

lr = LogisticRegression()
lr.fit(X_std,Y)
lr.predict(test_data_std)

Y


train_data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=None) 
X_train
Y_train
X_test
Y_test

#1 Decision Tree
#We dont need to use standart scaler for Decision Tree since distance doesnt
#matter here.

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="gini")

#dt=DecisionTreeClassifier()

dt = DecisionTreeClassifier(criterion="gini")
#dt = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)   


dt.fit(X_train,Y_train)
Y_pred=dt.predict(X_test)
Y_pred
Y_test
X_test



print("##############################################################################")


# Metric Results for Decision Tree Classifier
print("Decision Classifier Mean Squared Error: ",mean_squared_error(Y_test,Y_pred))
print("Decision Classifier Mean Absolute Error: ",mean_absolute_error(Y_test,Y_pred))
print("Decision Tree Classifier r2 Score: ",r2_score(Y_test,Y_pred))
print("Classification Report:\n ",classification_report(Y_test,Y_pred,digits = 2))
print("##############################################################################")

#check accuracy score
from sklearn.metrics import accuracy_score
dt_ac=accuracy_score(Y_test,Y_pred)

dt_ac


# #2 KNN 
# #We  need to use standart scaler for KNN since distance matter here.

X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)
X_test_std


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,metric="euclidean")

#knn=KNeighborsClassifier()

knn.fit(X_train_std,Y_train)

Y_pred=knn.predict(X_test_std)

Y_pred

Y_test

knn_ac=accuracy_score(Y_test,Y_pred)

knn_ac

# Metric Results for K-Nearest Neighbors
print("K-Nearest Neighbors Mean Squared Error: ",mean_squared_error(Y_test,Y_pred))
print("K-Nearest Neighbors Mean Absolute Error: ",mean_absolute_error(Y_test,Y_pred))
print("K-Nearest Neighbors r2 Score: ",r2_score(Y_test,Y_pred))
print("Classification Report:\n ",classification_report(Y_test,Y_pred,digits = 2))
print("##############################################################################")


#3.Algorithm → Logistic Regression

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear', random_state=0)


#lr=LogisticRegression()


lr.fit(X_train_std,Y_train)

Y_pred=lr.predict(X_test_std)
Y_pred
lr_ac=accuracy_score(Y_test,Y_pred)

lr_ac


# Metric Results for Logistic Regression
print("Logistic Regression Mean Squared Error: ",mean_squared_error(Y_test,Y_pred))
print("Logistic Regression Mean Absolute Error: ",mean_absolute_error(Y_test,Y_pred))
print("Logistic Regression r2 Score: ",r2_score(Y_test,Y_pred))
print("Classification Report:\n ",classification_report(Y_test,Y_pred,digits = 2))

plt.bar(x=['Decision Tree','KNN','Logistic Regression'],height=[dt_ac,knn_ac,lr_ac],color=['red', 'orange', 'blue']) #performance compare
plt.xlabel('\nAlgorithms')
plt.ylabel('Accuracy Score')
plt.show()

#Cross Validation Score
knn=KNeighborsClassifier(n_neighbors=5)
score=cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
print(score)
#mean of scores
print(score.mean())

# saving picture 
plt.savefig("compare.png", dpi=500, bbox_inches="tight")

